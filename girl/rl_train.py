"""
RL Training Script for Span Refinement Agent on VideoLights.

Usage:
    python -m girl.rl_train \
        --resume path/to/pretrained_videolights_ckpt.pth \
        --dset_name hl \
        --train_path data/highlight_train_release.jsonl \
        --eval_path data/highlight_val_release.jsonl \
        --v_feat_dirs ... --t_feat_dirs ... \
        --rl_epochs 50 --rl_lr 3e-4 --num_refine_steps 3 --max_action 0.1
"""

import os
import time
import json
import random
import logging
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from girl.config import BaseOptions
from girl.start_end_dataset import (
    StartEndDataset, start_end_collate, prepare_batch_inputs
)
from girl.inference import eval_epoch, setup_model, compute_mr_results, eval_epoch_post_processing
from girl.span_utils import span_cxw_to_xx, generalized_temporal_iou
from girl.matcher import HungarianMatcher
from girl.rl_agent import (
    SpanRefinementPolicy, DiscreteSpanPolicy, RLSpanRefiner,
    PPOTrainer, REINFORCETrainer, compute_iou_reward
)
from utils.basic_utils import AverageMeter
from standalone_eval.eval import eval_submission

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def add_rl_args(parser):
    """Add RL-specific arguments to the base argparser."""
    group = parser.add_argument_group("RL Refinement")
    group.add_argument("--rl_epochs", type=int, default=50, help="Number of RL training epochs")
    group.add_argument("--rl_lr", type=float, default=3e-4, help="RL agent learning rate")
    group.add_argument("--num_refine_steps", type=int, default=3, help="Number of iterative refinement steps")
    group.add_argument("--max_action", type=float, default=0.1, help="Max delta per refinement step")
    group.add_argument("--ppo_epochs", type=int, default=4, help="PPO inner update epochs")
    group.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO clip epsilon")
    group.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy bonus coefficient")
    group.add_argument("--value_coef", type=float, default=0.5, help="Value loss coefficient")
    group.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    group.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    group.add_argument("--rl_save_dir", type=str, default="results/rl_refine", help="RL results directory")
    group.add_argument("--rl_eval_every", type=int, default=5, help="Evaluate every N epochs")
    # Discrete policy flags (TSP-PRL inspired)
    group.add_argument("--discrete_actions", action="store_true", default=False,
                       help="Use discrete action space instead of continuous")
    group.add_argument("--use_gru", action="store_true", default=False,
                       help="Use GRU memory across refinement steps (discrete only)")
    group.add_argument("--use_gate_attn", action="store_true", default=False,
                       help="Use gate attention (text gates video features)")
    group.add_argument("--use_iou_aux", action="store_true", default=False,
                       help="Use IoU auxiliary prediction head (discrete only)")
    group.add_argument("--iou_aux_coef", type=float, default=0.5,
                       help="IoU auxiliary loss coefficient")
    # Reward shaping
    group.add_argument("--reward_scale", type=float, default=10.0,
                       help="Scale factor for delta-IoU reward")
    group.add_argument("--direction_coef", type=float, default=5.0,
                       help="Coefficient for direction-based reward (boundary distance reduction)")
    group.add_argument("--terminal_bonus", type=float, default=5.0,
                       help="Bonus reward at last step for overall IoU improvement")
    # Supervised pre-training (Phase 1)
    group.add_argument("--pretrain_epochs", type=int, default=10,
                       help="Number of supervised pre-training epochs (0 to skip)")
    group.add_argument("--pretrain_lr", type=float, default=1e-3,
                       help="Learning rate for supervised pre-training")
    # RL algorithm selection
    group.add_argument("--rl_algo", type=str, default="reinforce", choices=["reinforce", "ppo"],
                       help="RL algorithm: 'reinforce' (simpler) or 'ppo'")
    # Combined SL+RL training
    group.add_argument("--sl_coef", type=float, default=0.1,
                       help="Coefficient for supervised oracle loss mixed into RL phase (0 to disable)")
    # Curriculum
    group.add_argument("--curriculum", action="store_true", default=False,
                       help="Use curriculum: prioritize low-IoU (hard) predictions")
    group.add_argument("--iou_filter_thresh", type=float, default=0.9,
                       help="Only refine predictions with base IoU below this threshold")
    return parser


def match_predictions_to_targets(pred_spans, targets, matcher):
    """
    Use Hungarian matcher to assign predicted spans to ground truth spans.

    Args:
        pred_spans: (batch, num_queries, 2) - predicted (center, width)
        targets: dict with 'span_labels' list of dicts with 'spans' key
        matcher: HungarianMatcher instance

    Returns:
        matched_pred_idx: (total_matched,) - flat indices into pred_spans for matched queries
        matched_gt_spans: (total_matched, 2) - matched ground truth spans in (center, width)
        batch_query_indices: list of (batch_idx, query_idx) tuples
    """
    # Build dummy outputs dict for matcher
    outputs = {
        'pred_spans': pred_spans,
        'pred_logits': torch.zeros(pred_spans.shape[0], pred_spans.shape[1], 2, device=pred_spans.device),
    }
    target_dict = {'span_labels': targets['span_labels']}
    indices = matcher(outputs, target_dict)

    matched_pred_spans = []
    matched_gt_spans = []
    batch_indices = []
    query_indices = []

    for b_idx, (pred_idx, tgt_idx) in enumerate(indices):
        for p, t in zip(pred_idx, tgt_idx):
            matched_pred_spans.append(pred_spans[b_idx, p])
            matched_gt_spans.append(targets['span_labels'][b_idx]['spans'][t])
            batch_indices.append(b_idx)
            query_indices.append(p.item())

    if len(matched_pred_spans) == 0:
        return None, None, None, None

    matched_pred_spans = torch.stack(matched_pred_spans)  # (M, 2)
    matched_gt_spans = torch.stack(matched_gt_spans)       # (M, 2)

    return matched_pred_spans, matched_gt_spans, batch_indices, query_indices


def supervised_pretrain_epoch(rl_refiner, optimizer, matcher, train_loader, opt, epoch_i):
    """
    Phase 1: Supervised pre-training epoch.

    For each batch:
        1. Get base model predictions
        2. Match predictions to GT via Hungarian matching
        3. Build state for matched queries
        4. Compute oracle actions (which discrete action moves boundary closest to GT)
        5. Train policy with cross-entropy loss toward oracle actions

    This gives the policy a warm start so it already knows how to refine boundaries
    before RL fine-tuning. Inspired by TSP-PRL (Wu et al., AAAI 2020).
    """
    rl_refiner.base_model.eval()
    rl_refiner.policy.train()

    loss_meters = defaultdict(AverageMeter)

    for batch_idx, batch in tqdm(enumerate(train_loader), desc=f"SL Pretrain Epoch {epoch_i+1}",
                                  total=len(train_loader)):
        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)

        if targets is None or 'span_labels' not in targets:
            continue

        with torch.no_grad():
            base_outputs = rl_refiner.get_base_predictions(model_inputs)

        pred_spans = base_outputs['pred_spans'].detach()
        pred_logits = base_outputs['pred_logits'].detach()
        pred_scores = F.softmax(pred_logits, dim=-1)[..., 0]

        vid_mem = base_outputs['src_vid'].detach()
        txt_mem = base_outputs['src_txt'].detach()
        vid_mask = model_inputs['src_vid_mask']
        txt_mask = model_inputs['src_txt_mask']

        bsz, nq, _ = pred_spans.shape

        # Match predictions to GT
        matched_pred, matched_gt, b_indices, q_indices = match_predictions_to_targets(
            pred_spans, targets, matcher
        )
        if matched_pred is None:
            continue

        M = matched_pred.shape[0]
        flat_indices = [b * nq + q for b, q in zip(b_indices, q_indices)]
        flat_indices = torch.tensor(flat_indices, device=opt.device, dtype=torch.long)

        # Multi-step supervised training: at each step, build state from current span,
        # compute oracle action toward GT, and update
        current_spans = pred_spans.clone()
        hidden = None
        total_loss = torch.tensor(0.0, device=opt.device)
        num_steps = rl_refiner.num_refine_steps

        for step in range(num_steps):
            state = rl_refiner.build_state(
                current_spans, pred_scores,
                vid_mem, txt_mem, vid_mask, txt_mask
            )

            matched_state = state[flat_indices]
            matched_current = current_spans.view(bsz * nq, 2)[flat_indices]

            step_loss, info, hidden = rl_refiner.policy.supervised_loss(
                matched_state, matched_current, matched_gt, hidden=hidden, step=step
            )
            total_loss = total_loss + step_loss

            # Apply oracle actions to advance current_spans for next step
            with torch.no_grad():
                oracle_start, oracle_end = rl_refiner.policy.compute_oracle_actions(
                    matched_current, matched_gt, step=step
                )
                oracle_action = torch.stack([oracle_start, oracle_end], dim=-1)
                delta = rl_refiner.policy.action_to_delta(oracle_action, step=step)
                # Update matched spans
                new_matched = matched_current + delta
                new_matched[:, 0].clamp_(0.0, 1.0)
                new_matched[:, 1].clamp_(0.01, 1.0)
                # Write back to current_spans
                flat_current = current_spans.view(bsz * nq, 2)
                flat_current[flat_indices] = new_matched
                current_spans = flat_current.view(bsz, nq, 2)

            for k, v in info.items():
                loss_meters[k].update(v)

        avg_loss = total_loss / num_steps

        optimizer.zero_grad()
        avg_loss.backward()
        nn.utils.clip_grad_norm_(rl_refiner.policy.parameters(), 0.5)
        optimizer.step()

        loss_meters['total_loss'].update(avg_loss.item())

        if opt.debug and batch_idx == 3:
            break

    logger.info(f"[SL Pretrain Epoch {epoch_i+1}] "
                f"Loss: {loss_meters['total_loss'].avg:.4f}, "
                f"Start Loss: {loss_meters['start_loss'].avg:.4f}, "
                f"End Loss: {loss_meters['end_loss'].avg:.4f}, "
                f"Start Acc: {loss_meters['start_acc'].avg:.3f}, "
                f"End Acc: {loss_meters['end_acc'].avg:.3f}, "
                f"NoChange Frac: {loss_meters.get('no_change_frac', AverageMeter()).avg:.3f}")

    return {k: v.avg for k, v in loss_meters.items()}


def rl_train_epoch(rl_refiner, ppo_trainer, matcher, train_loader, opt, epoch_i):
    """
    One epoch of RL training.

    For each batch:
        1. Run frozen base model to get initial predictions
        2. Match predictions to ground truth via Hungarian matching
        3. Run multi-step refinement with RL agent
        4. Compute IoU rewards
        5. PPO update
    """
    rl_refiner.base_model.eval()
    rl_refiner.policy.train()

    loss_meters = defaultdict(AverageMeter)
    iou_meters = defaultdict(AverageMeter)

    for batch_idx, batch in tqdm(enumerate(train_loader), desc=f"RL Epoch {epoch_i+1}",
                                  total=len(train_loader)):
        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)

        if targets is None or 'span_labels' not in targets:
            continue

        # Forward through RL refiner
        rl_outputs = rl_refiner.forward_rl(model_inputs, targets)

        pred_spans = rl_outputs['initial_spans']     # (bsz, nq, 2)
        refined_spans = rl_outputs['refined_spans']  # (bsz, nq, 2)
        all_spans = rl_outputs['all_spans']          # list of (bsz*nq, 2)
        all_actions = rl_outputs['all_actions']
        all_log_probs = rl_outputs['all_log_probs']
        all_entropies = rl_outputs['all_entropies']
        all_values = rl_outputs['all_values']

        bsz, nq, _ = pred_spans.shape

        # Match predictions to ground truth (using initial predictions)
        matched_pred, matched_gt, b_indices, q_indices = match_predictions_to_targets(
            pred_spans, targets, matcher
        )

        if matched_pred is None:
            continue

        M = matched_pred.shape[0]

        # Build flat index: for each matched pair, find its position in the flattened (bsz*nq) tensor
        flat_indices = [b * nq + q for b, q in zip(b_indices, q_indices)]
        flat_indices = torch.tensor(flat_indices, device=opt.device, dtype=torch.long)

        # Extract matched spans at each refinement step
        matched_all_spans = [s[flat_indices] for s in all_spans]  # list of (M, 2)
        matched_actions = [a[flat_indices] for a in all_actions]
        matched_log_probs = [lp[flat_indices] for lp in all_log_probs]
        matched_entropies = [e[flat_indices] for e in all_entropies]
        matched_values = [v[flat_indices] for v in all_values]

        # Compute rewards based on IoU improvement
        rewards = ppo_trainer.compute_rewards(
            matched_all_spans, matched_gt, rl_refiner.num_refine_steps
        )

        # Log reward statistics periodically
        if batch_idx % 20 == 0:
            with torch.no_grad():
                r_mean = torch.stack(rewards).mean().item()
                r_abs = torch.stack(rewards).abs().mean().item()
                act_sample = matched_actions[0][:3].tolist() if matched_actions else []
                logger.info(f"  [batch {batch_idx}] reward_mean={r_mean:.5f} "
                            f"reward_abs_mean={r_abs:.5f} actions_sample={act_sample}")

        # Build states for matched queries at each step
        # We need to rebuild states for matched queries only
        vid_mem = rl_outputs['base_outputs']['src_vid'].detach()
        txt_mem = rl_outputs['base_outputs']['src_txt'].detach()
        vid_mask = model_inputs['src_vid_mask']
        txt_mask = model_inputs['src_txt_mask']
        pred_scores = rl_outputs['pred_scores']

        matched_states = []
        for step in range(rl_refiner.num_refine_steps):
            current_spans = all_spans[step]  # (bsz*nq, 2)
            current_spans_reshaped = current_spans.view(bsz, nq, 2)
            state = rl_refiner.build_state(
                current_spans_reshaped, pred_scores,
                vid_mem, txt_mem, vid_mask, txt_mask
            )
            matched_states.append(state[flat_indices].detach())

        # Prepare IoU auxiliary data if available
        all_iou_preds = rl_outputs.get('all_iou_preds', [None] * rl_refiner.num_refine_steps)
        matched_iou_preds = None
        matched_iou_targets = None
        if all_iou_preds[0] is not None:
            matched_iou_preds = [p[flat_indices] for p in all_iou_preds]
            # IoU targets: actual IoU of refined spans at each step
            matched_iou_targets = []
            for step in range(rl_refiner.num_refine_steps):
                step_iou = compute_iou_reward(matched_all_spans[step + 1], matched_gt)
                matched_iou_targets.append(step_iou)

        # PPO update on matched queries
        loss_info = ppo_trainer.update(
            matched_states, matched_actions, matched_log_probs,
            rewards, matched_values, matched_entropies,
            iou_targets_list=matched_iou_targets,
            iou_preds_list=matched_iou_preds,
        )

        # Log metrics
        for k, v in loss_info.items():
            loss_meters[k].update(v)

        # Compute IoU before and after refinement
        with torch.no_grad():
            iou_before = compute_iou_reward(matched_all_spans[0], matched_gt).mean().item()
            iou_after = compute_iou_reward(matched_all_spans[-1], matched_gt).mean().item()
            iou_meters['iou_before'].update(iou_before, M)
            iou_meters['iou_after'].update(iou_after, M)
            iou_meters['iou_improvement'].update(iou_after - iou_before, M)

        if opt.debug and batch_idx == 3:
            break

    # Print epoch summary
    is_discrete = hasattr(rl_refiner, 'is_discrete') and rl_refiner.is_discrete
    logger.info(f"[RL Epoch {epoch_i+1}] discrete={is_discrete} "
                f"Policy Loss: {loss_meters['policy_loss'].avg:.4f}, "
                f"Value Loss: {loss_meters['value_loss'].avg:.4f}, "
                f"IoU Before: {iou_meters['iou_before'].avg:.4f}, "
                f"IoU After: {iou_meters['iou_after'].avg:.4f}, "
                f"IoU Improvement: {iou_meters['iou_improvement'].avg:.4f}")

    # Log action distribution for discrete policies
    if is_discrete:
        with torch.no_grad():
            dummy_state = torch.randn(100, rl_refiner.policy.shared[0].in_features,
                                       device=next(rl_refiner.policy.parameters()).device)
            s_logits, e_logits, _, _, _ = rl_refiner.policy(dummy_state)
            s_dist = torch.softmax(s_logits, dim=-1).mean(dim=0)
            e_dist = torch.softmax(e_logits, dim=-1).mean(dim=0)
            logger.info(f"  Action probs (start): {[f'{p:.3f}' for p in s_dist.tolist()]}")
            logger.info(f"  Action probs (end):   {[f'{p:.3f}' for p in e_dist.tolist()]}")

    return {k: v.avg for k, v in {**loss_meters, **iou_meters}.items()}


def reinforce_train_epoch(rl_refiner, reinforce_trainer, matcher, train_loader, opt, epoch_i):
    """
    One epoch of REINFORCE training with trajectory-level IoU reward.

    Simpler than PPO and more stable for sparse reward settings.
    Uses final IoU as reward with exponential moving average baseline.
    """
    rl_refiner.base_model.eval()
    rl_refiner.policy.train()

    loss_meters = defaultdict(AverageMeter)
    iou_meters = defaultdict(AverageMeter)

    iou_filter = getattr(opt, 'iou_filter_thresh', 1.0)
    use_curriculum = getattr(opt, 'curriculum', False)

    for batch_idx, batch in tqdm(enumerate(train_loader), desc=f"REINFORCE Epoch {epoch_i+1}",
                                  total=len(train_loader)):
        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)

        if targets is None or 'span_labels' not in targets:
            continue

        rl_outputs = rl_refiner.forward_rl(model_inputs, targets)

        pred_spans = rl_outputs['initial_spans']
        all_spans = rl_outputs['all_spans']
        all_log_probs = rl_outputs['all_log_probs']
        all_entropies = rl_outputs['all_entropies']
        all_iou_preds = rl_outputs.get('all_iou_preds', [None] * rl_refiner.num_refine_steps)

        bsz, nq, _ = pred_spans.shape

        matched_pred, matched_gt, b_indices, q_indices = match_predictions_to_targets(
            pred_spans, targets, matcher
        )
        if matched_pred is None:
            continue

        M = matched_pred.shape[0]
        flat_indices = [b * nq + q for b, q in zip(b_indices, q_indices)]
        flat_indices = torch.tensor(flat_indices, device=opt.device, dtype=torch.long)

        # Extract matched data
        matched_all_spans = [s[flat_indices] for s in all_spans]
        matched_log_probs = [lp[flat_indices] for lp in all_log_probs]
        matched_entropies = [e[flat_indices] for e in all_entropies]

        # Curriculum: filter to low-IoU samples where refinement has more room
        if use_curriculum and iou_filter < 1.0:
            with torch.no_grad():
                base_iou = compute_iou_reward(matched_all_spans[0], matched_gt)
                keep_mask = base_iou < iou_filter
                if keep_mask.sum() == 0:
                    continue
                matched_all_spans = [s[keep_mask] for s in matched_all_spans]
                matched_log_probs = [lp[keep_mask] for lp in matched_log_probs]
                matched_entropies = [e[keep_mask] for e in matched_entropies]
                matched_gt_filtered = matched_gt[keep_mask]
                flat_indices_filtered = flat_indices[keep_mask]
                M = keep_mask.sum().item()
        else:
            matched_gt_filtered = matched_gt
            flat_indices_filtered = flat_indices

        # Trajectory reward: final IoU
        reward, improvement = reinforce_trainer.compute_trajectory_reward(
            matched_all_spans, matched_gt_filtered, rl_refiner.num_refine_steps
        )

        # IoU aux data
        matched_iou_preds = None
        matched_iou_targets = None
        if all_iou_preds[0] is not None:
            matched_iou_preds = [p[flat_indices_filtered] for p in all_iou_preds]
            matched_iou_targets = []
            for step in range(rl_refiner.num_refine_steps):
                step_iou = compute_iou_reward(matched_all_spans[step + 1], matched_gt_filtered)
                matched_iou_targets.append(step_iou)

        # REINFORCE update
        loss_info = reinforce_trainer.update(
            matched_log_probs, matched_entropies, reward,
            iou_targets_list=matched_iou_targets,
            iou_preds_list=matched_iou_preds,
        )

        # Combined SL+RL: add supervised oracle loss to guide exploration
        sl_coef = getattr(opt, 'sl_coef', 0.0)
        if sl_coef > 0 and rl_refiner.is_discrete:
            vid_mem = rl_outputs['base_outputs']['src_vid'].detach()
            txt_mem = rl_outputs['base_outputs']['src_txt'].detach()
            vid_mask = model_inputs['src_vid_mask']
            txt_mask = model_inputs['src_txt_mask']
            pred_scores = rl_outputs['pred_scores']

            sl_loss_total = torch.tensor(0.0, device=opt.device)
            for step in range(rl_refiner.num_refine_steps):
                current_s = all_spans[step].view(bsz, nq, 2)
                state = rl_refiner.build_state(
                    current_s, pred_scores, vid_mem, txt_mem, vid_mask, txt_mask
                )
                m_state = state[flat_indices_filtered]
                m_current = all_spans[step][flat_indices_filtered]
                sl_step_loss, _, _ = rl_refiner.policy.supervised_loss(
                    m_state, m_current, matched_gt_filtered, step=step
                )
                sl_loss_total = sl_loss_total + sl_step_loss
            sl_loss_avg = sl_loss_total / rl_refiner.num_refine_steps

            reinforce_trainer.optimizer.zero_grad()
            (sl_coef * sl_loss_avg).backward()
            nn.utils.clip_grad_norm_(rl_refiner.policy.parameters(), 0.5)
            reinforce_trainer.optimizer.step()
            loss_info['sl_loss'] = sl_loss_avg.item()

        for k, v in loss_info.items():
            loss_meters[k].update(v)

        with torch.no_grad():
            iou_before = compute_iou_reward(matched_all_spans[0], matched_gt_filtered).mean().item()
            iou_after = compute_iou_reward(matched_all_spans[-1], matched_gt_filtered).mean().item()
            iou_meters['iou_before'].update(iou_before, M)
            iou_meters['iou_after'].update(iou_after, M)
            iou_meters['iou_improvement'].update(iou_after - iou_before, M)

        if batch_idx % 20 == 0:
            logger.info(f"  [batch {batch_idx}] reward_mean={reward.mean().item():.4f} "
                        f"baseline={loss_info.get('baseline', 0):.4f} "
                        f"advantage_mean={loss_info.get('advantage_mean', 0):.4f}")

        if opt.debug and batch_idx == 3:
            break

    is_discrete = hasattr(rl_refiner, 'is_discrete') and rl_refiner.is_discrete
    logger.info(f"[REINFORCE Epoch {epoch_i+1}] discrete={is_discrete} "
                f"Policy Loss: {loss_meters['policy_loss'].avg:.4f}, "
                f"IoU Before: {iou_meters['iou_before'].avg:.4f}, "
                f"IoU After: {iou_meters['iou_after'].avg:.4f}, "
                f"IoU Improvement: {iou_meters['iou_improvement'].avg:.4f}, "
                f"Baseline: {loss_meters.get('baseline', AverageMeter()).avg:.4f}")

    if is_discrete:
        with torch.no_grad():
            dummy_state = torch.randn(100, rl_refiner.policy.shared[0].in_features,
                                       device=next(rl_refiner.policy.parameters()).device)
            s_logits, e_logits, _, _, _ = rl_refiner.policy(dummy_state)
            s_dist = torch.softmax(s_logits, dim=-1).mean(dim=0)
            e_dist = torch.softmax(e_logits, dim=-1).mean(dim=0)
            logger.info(f"  Action probs (start): {[f'{p:.3f}' for p in s_dist.tolist()]}")
            logger.info(f"  Action probs (end):   {[f'{p:.3f}' for p in e_dist.tolist()]}")

    return {k: v.avg for k, v in {**loss_meters, **iou_meters}.items()}


def rl_eval_epoch(rl_refiner, matcher, eval_loader, opt):
    """
    Evaluate RL-refined predictions.
    Computes IoU improvement over base model predictions.
    """
    rl_refiner.base_model.eval()
    rl_refiner.policy.eval()

    iou_meters = defaultdict(AverageMeter)

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(eval_loader), desc="RL Eval",
                                      total=len(eval_loader)):
            model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)

            if targets is None or 'span_labels' not in targets:
                continue

            rl_outputs = rl_refiner.forward_rl(model_inputs, targets, deterministic=True)

            pred_spans = rl_outputs['initial_spans']
            bsz, nq, _ = pred_spans.shape

            matched_pred, matched_gt, b_indices, q_indices = match_predictions_to_targets(
                pred_spans, targets, matcher
            )

            if matched_pred is None:
                continue

            M = matched_pred.shape[0]
            flat_indices = [b * nq + q for b, q in zip(b_indices, q_indices)]
            flat_indices = torch.tensor(flat_indices, device=opt.device, dtype=torch.long)

            all_spans = rl_outputs['all_spans']
            iou_before = compute_iou_reward(all_spans[0][flat_indices], matched_gt).mean().item()
            iou_after = compute_iou_reward(all_spans[-1][flat_indices], matched_gt).mean().item()

            iou_meters['iou_before'].update(iou_before, M)
            iou_meters['iou_after'].update(iou_after, M)
            iou_meters['iou_improvement'].update(iou_after - iou_before, M)

    logger.info(f"[RL Eval] "
                f"IoU Before: {iou_meters['iou_before'].avg:.4f}, "
                f"IoU After: {iou_meters['iou_after'].avg:.4f}, "
                f"IoU Improvement: {iou_meters['iou_improvement'].avg:.4f}")

    return {k: v.avg for k, v in iou_meters.items()}


class RLRefinedModelWrapper(nn.Module):
    """Wraps base model + RL policy so it can be used with existing eval pipeline (compute_mr_results)."""

    def __init__(self, rl_refiner):
        super().__init__()
        self.rl_refiner = rl_refiner
        # Expose hidden_dim for compatibility
        self.hidden_dim = rl_refiner.base_model.hidden_dim

    @torch.no_grad()
    def forward(self, **model_inputs):
        rl_outputs = self.rl_refiner.forward_rl(model_inputs, deterministic=True)
        base_outputs = rl_outputs['base_outputs']
        refined_outputs = dict(base_outputs)
        refined_outputs['pred_spans'] = rl_outputs['refined_spans']
        return refined_outputs


def rl_eval_standard_metrics(rl_refiner, eval_dataset, eval_loader, opt, epoch_i=None, tag="RL"):
    """
    Evaluate RL-refined predictions using standard metrics (R1@0.5, R1@0.7, mAP, etc.)
    Also evaluates base model for comparison.

    Returns:
        base_metrics: dict or None, standard metrics of base model
        rl_metrics: dict or None, standard metrics of RL-refined model
    """
    import torch.nn.functional as F

    rl_refiner.base_model.eval()
    rl_refiner.policy.eval()

    # Ensure results_dir exists
    results_dir = getattr(opt, 'results_dir', opt.rl_save_dir)
    os.makedirs(results_dir, exist_ok=True)
    saved_results_dir = opt.results_dir if hasattr(opt, 'results_dir') else None
    opt.results_dir = results_dir

    # Ensure eval_split_name is set
    if not hasattr(opt, 'eval_split_name'):
        opt.eval_split_name = "val"

    gt_data = eval_dataset.data

    # --- Evaluate base model (without RL) ---
    logger.info(f"[{tag}] Evaluating base model predictions...")
    base_submission, _ = compute_mr_results(rl_refiner.base_model, eval_loader, opt)
    # Strip saliency scores to only evaluate MR metrics
    for s in base_submission:
        s.pop('pred_saliency_scores', None)
    base_metrics_result = None
    if opt.eval_split_name == "val":
        base_metrics_result = eval_submission(base_submission, gt_data, verbose=False, match_number=False)
        brief = base_metrics_result.get("brief", base_metrics_result)
        logger.info(f"[{tag}] Base model metrics: {json.dumps(brief, indent=2)}")

    # --- Evaluate RL-refined model ---
    logger.info(f"[{tag}] Evaluating RL-refined predictions...")
    rl_wrapper = RLRefinedModelWrapper(rl_refiner)

    # Diagnostic: check if RL actually changes spans
    with torch.no_grad():
        sample_batch = next(iter(eval_loader))
        sample_inputs, _ = prepare_batch_inputs(sample_batch[1], opt.device, non_blocking=opt.pin_memory)
        rl_out = rl_refiner.forward_rl(sample_inputs, deterministic=True)
        base_spans = rl_out['initial_spans']
        refined_spans = rl_out['refined_spans']
        span_diff = (refined_spans - base_spans).abs()
        logger.info(f"[{tag}] Diagnostic: is_discrete={rl_refiner.is_discrete}, "
                     f"span_diff_max={span_diff.max().item():.6f}, "
                     f"span_diff_mean={span_diff.mean().item():.6f}, "
                     f"actions_sample={rl_out['all_actions'][0][:3].tolist()}")

    rl_submission, _ = compute_mr_results(rl_wrapper, eval_loader, opt)
    for s in rl_submission:
        s.pop('pred_saliency_scores', None)
    rl_metrics_result = None
    if opt.eval_split_name == "val":
        rl_metrics_result = eval_submission(rl_submission, gt_data, verbose=False, match_number=False)
        brief = rl_metrics_result.get("brief", rl_metrics_result)
        logger.info(f"[{tag}] RL-refined metrics: {json.dumps(brief, indent=2)}")

    # --- Side-by-side comparison ---
    if base_metrics_result and rl_metrics_result:
        base_brief = base_metrics_result.get("brief", {})
        rl_brief = rl_metrics_result.get("brief", {})
        logger.info(f"[{tag}] === Metric Comparison (Base vs RL) ===")
        for key in base_brief:
            b_val = base_brief.get(key, 0)
            r_val = rl_brief.get(key, 0)
            diff = r_val - b_val
            symbol = "+" if diff >= 0 else ""
            logger.info(f"  {key}: {b_val:.2f} -> {r_val:.2f} ({symbol}{diff:.2f})")

    if saved_results_dir is not None:
        opt.results_dir = saved_results_dir

    return base_metrics_result, rl_metrics_result


def train_rl(opt):
    """Main RL training function."""
    # Set seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed)

    # Create save directory
    os.makedirs(opt.rl_save_dir, exist_ok=True)

    # Load pretrained VideoLights model
    logger.info("Loading pretrained VideoLights model...")
    model, criterion, _, _ = setup_model(opt)
    model.eval()

    # Create RL policy
    hidden_dim = model.hidden_dim  # 256
    # State: span(2) + score(1) + vid_local(D) + start_feat(D) + end_feat(D) + txt(D)
    state_dim = 2 + 1 + hidden_dim * 4
    use_discrete = getattr(opt, 'discrete_actions', False)
    use_gru = getattr(opt, 'use_gru', False)
    use_gate_attn = getattr(opt, 'use_gate_attn', False)
    use_iou_aux = getattr(opt, 'use_iou_aux', False)
    iou_aux_coef = getattr(opt, 'iou_aux_coef', 0.0) if use_iou_aux else 0.0

    if use_discrete:
        policy = DiscreteSpanPolicy(
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            max_v_l=opt.max_v_l,
            use_gate_attn=use_gate_attn,
            use_gru=use_gru,
            use_iou_aux=use_iou_aux,
            num_refine_steps=opt.num_refine_steps,
        ).to(opt.device)
        logger.info(f"Using DISCRETE policy (GRU={use_gru}, gate_attn={use_gate_attn}, iou_aux={use_iou_aux})")
    else:
        policy = SpanRefinementPolicy(
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            max_action=opt.max_action,
        ).to(opt.device)
        logger.info("Using CONTINUOUS policy")

    logger.info(f"RL Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # Create RL refiner
    rl_refiner = RLSpanRefiner(
        base_model=model,
        policy=policy,
        num_refine_steps=opt.num_refine_steps,
        use_gate_attn=use_gate_attn,
    ).to(opt.device)

    # Create RL trainer based on algorithm choice
    rl_algo = getattr(opt, 'rl_algo', 'ppo')
    if rl_algo == 'reinforce':
        rl_trainer = REINFORCETrainer(
            policy=policy,
            lr=opt.rl_lr,
            entropy_coef=opt.entropy_coef,
            iou_aux_coef=iou_aux_coef,
        )
        logger.info("Using REINFORCE trainer")
    else:
        rl_trainer = PPOTrainer(
            policy=policy,
            lr=opt.rl_lr,
            gamma=opt.gamma,
            gae_lambda=opt.gae_lambda,
            clip_epsilon=opt.clip_epsilon,
            entropy_coef=opt.entropy_coef,
            value_coef=opt.value_coef,
            ppo_epochs=opt.ppo_epochs,
            iou_aux_coef=iou_aux_coef,
            reward_scale=getattr(opt, 'reward_scale', 10.0),
            direction_coef=getattr(opt, 'direction_coef', 5.0),
            terminal_bonus=getattr(opt, 'terminal_bonus', 5.0),
        )
        logger.info("Using PPO trainer")

    # Create matcher for prediction-target assignment
    matcher = HungarianMatcher(
        cost_class=opt.set_cost_class,
        cost_span=opt.set_cost_span,
        cost_giou=opt.set_cost_giou,
        span_loss_type=opt.span_loss_type,
        max_v_l=opt.max_v_l,
    )

    # Create datasets
    train_dataset = StartEndDataset(
        dset_name=opt.dset_name,
        data_path=opt.train_path,
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dirs=opt.t_feat_dirs,
        q_feat_type="last_hidden_state",
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        data_ratio=opt.data_ratio,
        normalize_v=not opt.no_norm_vfeat,
        normalize_t=not opt.no_norm_tfeat,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=0,  # no text dropout during RL training
    )

    eval_dataset = StartEndDataset(
        dset_name=opt.dset_name,
        data_path=opt.eval_path,
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dirs=opt.t_feat_dirs,
        q_feat_type="last_hidden_state",
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        data_ratio=opt.data_ratio,
        normalize_v=not opt.no_norm_vfeat,
        normalize_t=not opt.no_norm_tfeat,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=0,
    )

    train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
        pin_memory=opt.pin_memory,
    )

    eval_loader = DataLoader(
        eval_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.eval_bsz,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=opt.pin_memory,
    )

    # Training loop
    best_iou_improvement = -float('inf')
    best_epoch = -1

    # ===== Phase 1: Supervised Pre-training =====
    pretrain_epochs = getattr(opt, 'pretrain_epochs', 0)
    if pretrain_epochs > 0:
        logger.info(f"===== Phase 1: Supervised Pre-training ({pretrain_epochs} epochs) =====")
        pretrain_lr = getattr(opt, 'pretrain_lr', 1e-3)
        pretrain_optimizer = torch.optim.Adam(
            rl_refiner.policy.parameters(), lr=pretrain_lr, weight_decay=1e-4
        )
        pretrain_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            pretrain_optimizer, T_max=pretrain_epochs
        )

        for pt_epoch in range(pretrain_epochs):
            pt_metrics = supervised_pretrain_epoch(
                rl_refiner, pretrain_optimizer, matcher, train_loader, opt, pt_epoch
            )
            pretrain_scheduler.step()

            # Evaluate after pretraining
            if (pt_epoch + 1) % max(1, pretrain_epochs // 3) == 0 or pt_epoch == pretrain_epochs - 1:
                eval_metrics = rl_eval_epoch(rl_refiner, matcher, eval_loader, opt)
                logger.info(f"[Pretrain Eval Epoch {pt_epoch+1}] IoU Improvement: {eval_metrics.get('iou_improvement', 0):.4f}")

        # Save pretrained policy
        pt_ckpt = os.path.join(opt.rl_save_dir, "pretrained_policy.pth")
        torch.save({
            'policy_state_dict': policy.state_dict(),
            'pretrain_epochs': pretrain_epochs,
        }, pt_ckpt)
        logger.info(f"Saved pretrained policy to {pt_ckpt}")

    # ===== Phase 2: RL Fine-tuning =====
    logger.info(f"===== Phase 2: RL Fine-tuning ({opt.rl_epochs} epochs, algo={rl_algo}) =====")
    for epoch_i in range(opt.rl_epochs):
        if rl_algo == 'reinforce':
            train_metrics = reinforce_train_epoch(
                rl_refiner, rl_trainer, matcher, train_loader, opt, epoch_i
            )
        else:
            train_metrics = rl_train_epoch(
                rl_refiner, rl_trainer, matcher, train_loader, opt, epoch_i
            )

        # Evaluate periodically
        if (epoch_i + 1) % opt.rl_eval_every == 0 or epoch_i == opt.rl_epochs - 1:
            eval_metrics = rl_eval_epoch(rl_refiner, matcher, eval_loader, opt)

            # Standard metrics evaluation (R1@0.5, R1@0.7, mAP, etc.)
            base_metrics, rl_metrics = rl_eval_standard_metrics(
                rl_refiner, eval_dataset, eval_loader, opt,
                epoch_i=epoch_i, tag=f"Epoch {epoch_i+1}"
            )

            iou_imp = eval_metrics.get('iou_improvement', 0)
            if iou_imp > best_iou_improvement:
                best_iou_improvement = iou_imp
                best_epoch = epoch_i + 1

                # Save best RL policy checkpoint
                ckpt_path = os.path.join(opt.rl_save_dir, "best_rl_policy.pth")
                torch.save({
                    'epoch': epoch_i + 1,
                    'policy_state_dict': policy.state_dict(),
                    'optimizer_state_dict': rl_trainer.optimizer.state_dict(),
                    'iou_improvement': iou_imp,
                    'eval_metrics': eval_metrics,
                    'rl_config': {
                        'hidden_dim': hidden_dim,
                        'state_dim': state_dim,
                        'max_action': getattr(opt, 'max_action', 0.1),
                        'num_refine_steps': opt.num_refine_steps,
                        'discrete_actions': use_discrete,
                        'use_gru': use_gru,
                        'use_gate_attn': use_gate_attn,
                        'use_iou_aux': use_iou_aux,
                    },
                }, ckpt_path)
                logger.info(f"Saved best RL policy at epoch {epoch_i+1} with IoU improvement: {iou_imp:.4f}")

        # Save latest checkpoint
        latest_path = os.path.join(opt.rl_save_dir, "latest_rl_policy.pth")
        torch.save({
            'epoch': epoch_i + 1,
            'policy_state_dict': policy.state_dict(),
            'optimizer_state_dict': rl_trainer.optimizer.state_dict(),
            'train_metrics': train_metrics,
        }, latest_path)

    logger.info(f"RL Training complete. Best epoch: {best_epoch}, "
                f"Best IoU improvement: {best_iou_improvement:.4f}")

    return rl_refiner


if __name__ == "__main__":
    # Initialize BaseOptions and add RL args before parsing
    base_opt = BaseOptions()
    base_opt.initialize()
    add_rl_args(base_opt.parser)

    # Use BaseOptions.parse() to get proper post-processing (v_feat_dim += 2 for tef, device setup, etc.)
    # But we need to monkey-patch the parser so parse() uses our extended version
    opt = base_opt.parser.parse_args()

    # Replicate BaseOptions.parse() post-processing
    if opt.debug:
        opt.results_root = os.path.sep.join(opt.results_root.split(os.path.sep)[:-1] + ["debug_results"])
        opt.num_workers = 0

    opt.device = torch.device(f"cuda:{opt.device}" if isinstance(opt.device, int) and opt.device >= 0 else "cpu")
    if opt.device.type == "cuda":
        torch.cuda.set_device(opt.device)
    opt.pin_memory = not getattr(opt, 'no_pin_memory', False)

    opt.use_tef = "tef" in opt.ctx_mode
    opt.use_video = "video" in opt.ctx_mode
    if not opt.use_video:
        opt.v_feat_dim = 0
    if opt.use_tef:
        opt.v_feat_dim += 2

    if not hasattr(opt, 'no_norm_vfeat'):
        opt.no_norm_vfeat = False
    if not hasattr(opt, 'no_norm_tfeat'):
        opt.no_norm_tfeat = False
    if not hasattr(opt, 'no_sort_results'):
        opt.no_sort_results = False
    if not hasattr(opt, 'results_dir'):
        opt.results_dir = opt.rl_save_dir
    if not hasattr(opt, 'eval_split_name'):
        opt.eval_split_name = "val"

    train_rl(opt)
