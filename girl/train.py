import os
import time
import json
import pprint
import random
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

from girl.config import BaseOptions
from girl.start_end_dataset import \
    StartEndDataset, start_end_collate, prepare_batch_inputs
# from qd_detr.start_end_dataset_audio import \
#     StartEndDataset_audio, start_end_collate_audio, prepare_batch_inputs_audio
from girl.inference import (
    build_finetune_prefixes,
    configure_frozen_backbone,
    eval_epoch,
    matches_prefix,
    start_inference,
    setup_model,
)
from girl.span_utils import temporal_iou, span_cxw_to_xx, generalized_temporal_iou
from utils.basic_utils import AverageMeter, dict_to_markdown
from utils.model_utils import count_parameters
from fvcore.nn import FlopCountAnalysis


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


class ModelEMA:
    """Exponential Moving Average of model parameters.
    Maintains a shadow copy of model weights updated as:
        shadow = decay * shadow + (1 - decay) * model_params
    Use the shadow weights for evaluation to get smoother predictions.
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)
        # Also update buffers (batch norm running stats, etc.)
        for ema_b, model_b in zip(self.shadow.buffers(), model.buffers()):
            ema_b.data.copy_(model_b.data)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer, ema=None):
    logger.info(f"[Epoch {epoch_i+1}]")
    model.train()
    criterion.train()

    # init meters
    time_meters = defaultdict(AverageMeter)
    loss_meters = defaultdict(AverageMeter)

    num_training_examples = len(train_loader)
    timer_dataloading = time.time()
    for batch_idx, batch in tqdm(enumerate(train_loader),
                                 desc="Training Iteration",
                                 total=num_training_examples):
        time_meters["dataloading_time"].update(time.time() - timer_dataloading)

        timer_start = time.time()
        # if opt.a_feat_dir is None:
        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)
        # else:
        #     model_inputs, targets = prepare_batch_inputs_audio(batch[1], opt.device, non_blocking=opt.pin_memory)
        time_meters["prepare_inputs_time"].update(time.time() - timer_start)
        timer_start = time.time()
        outputs = model(**model_inputs)
        loss_dict = criterion(outputs, targets, epoch_i)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        time_meters["model_forward_time"].update(time.time() - timer_start)

        timer_start = time.time()
        optimizer.zero_grad()
        losses.backward()
        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        time_meters["model_backward_time"].update(time.time() - timer_start)

        loss_dict["loss_overall"] = float(losses)  # for logging only
        for k, v in loss_dict.items():
            loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

        timer_dataloading = time.time()
        if opt.debug and batch_idx == 3:
            break

    # print/add logs
    tb_writer.add_scalar("Train/lr", float(optimizer.param_groups[0]["lr"]), epoch_i+1)
    for k, v in loss_meters.items():
        tb_writer.add_scalar("Train/{}".format(k), v.avg, epoch_i+1)

    to_write = opt.train_log_txt_formatter.format(
        time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
        epoch=epoch_i+1,
        loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()]))
    with open(opt.train_log_filepath, "a") as f:
        f.write(to_write)

    logger.info("Epoch time stats:")
    for name, meter in time_meters.items():
        d = {k: f"{getattr(meter, k):.4f}" for k in ["max", "min", "avg"]}
        logger.info(f"{name} ==> {d}")


def _compute_query_ious(pred_spans_cxw, gt_spans_cxw):
    """Compute max IoU for each query against all GT spans.
    Args:
        pred_spans_cxw: (n_queries, 2) center-width, normalized
        gt_spans_cxw: (n_gt, 2) center-width, normalized
    Returns:
        ious: (n_queries,) max IoU per query
    """
    pred_xx = span_cxw_to_xx(pred_spans_cxw)
    gt_xx = span_cxw_to_xx(gt_spans_cxw)
    # Broadcast: pred (nq,1,2) vs gt (1,ngt,2)
    pred_s = pred_xx[:, 0].unsqueeze(1)  # (nq, 1)
    pred_e = pred_xx[:, 1].unsqueeze(1)
    gt_s = gt_xx[:, 0].unsqueeze(0)      # (1, ngt)
    gt_e = gt_xx[:, 1].unsqueeze(0)
    inter_s = torch.max(pred_s, gt_s)
    inter_e = torch.min(pred_e, gt_e)
    inter = (inter_e - inter_s).clamp(min=0)
    pred_len = (pred_e - pred_s).clamp(min=1e-6)
    gt_len = (gt_e - gt_s).clamp(min=1e-6)
    union = pred_len + gt_len - inter
    iou_mat = inter / union.clamp(min=1e-6)  # (nq, ngt)
    return iou_mat.max(dim=1).values  # (nq,)


def score_iou_ranking_loss(outputs, targets, margin=0.1, iou_threshold=0.1):
    """Pairwise ranking loss to align foreground scores with IoU.

    For each query pair (i, j) where IoU_i > IoU_j + iou_threshold:
        loss += max(0, margin - (score_i - score_j))

    This directly trains the model's foreground score to correlate with IoU,
    so that top-1 by score ≈ top-1 by IoU at eval time.
    """
    pred_logits = outputs['pred_logits']  # (bsz, n_queries, 2)
    pred_spans = outputs['pred_spans']    # (bsz, n_queries, 2)
    prob = F.softmax(pred_logits, dim=-1)
    query_scores = prob[..., 0]  # (bsz, n_queries) foreground scores

    span_labels = targets["span_labels"]
    total_loss = torch.tensor(0.0, device=pred_logits.device)
    n_valid = 0

    for i in range(len(span_labels)):
        gt_cxw = span_labels[i]["spans"]
        if gt_cxw.shape[0] == 0:
            continue

        with torch.no_grad():
            ious = _compute_query_ious(pred_spans[i].detach(), gt_cxw)  # (n_queries,)

        scores_i = query_scores[i]  # (n_queries,) WITH gradient
        # Pairwise: IoU_a - IoU_b > threshold  →  score_a should be > score_b
        iou_diff = ious.unsqueeze(1) - ious.unsqueeze(0)   # (nq, nq) detached
        score_diff = scores_i.unsqueeze(1) - scores_i.unsqueeze(0)  # (nq, nq) with grad

        should_rank_higher = (iou_diff > iou_threshold)  # (nq, nq) bool mask
        if not should_rank_higher.any():
            continue

        pairwise_loss = torch.clamp(margin - score_diff, min=0)  # (nq, nq)
        n_pairs = should_rank_higher.float().sum().clamp(min=1)
        loss_i = (pairwise_loss * should_rank_higher.float()).sum() / n_pairs

        total_loss = total_loss + loss_i
        n_valid += 1

    return total_loss / max(n_valid, 1)


def scst_rl_loss(outputs, targets, span_loss_type="l1", temperature=0.5, n_samples=3,
                 decompose=False, threshold_aware=False, span_weight_mode="improvement",
                 use_giou=False, boundary_bonus=False, threshold_steps=False,
                 iou_greedy=False, baseline_mode="score_greedy",
                 reward_mode="iou", clip_adv=0.0,
                 ic_reward_weight=0.0, cendist_weight=0.0, top_k=3):
    """Self-Critical Sequence Training loss for moment retrieval.

    Two components:
    1. Score REINFORCE: policy gradient on query ranking scores to prefer
       high-IoU queries. Backprops through pred_logits and pred_iou.
    2. Reward-weighted span regression: direct L1 push on span boundaries
       weighted by how much each query could improve. Backprops through
       pred_spans. This is the key difference from v1 which detached spans.

    Args:
        decompose: If True, return (score_reinforce_loss, span_regression_loss) separately.
                   If False, return their sum (default, backward compatible).

    References:
      - Rennie et al., CVPR 2017 (SCST for captioning)
      - Wu et al., AAAI 2020 (TSP-PRL for temporal grounding)
    """
    pred_logits = outputs['pred_logits']  # (bsz, #queries, 2)
    pred_spans = outputs['pred_spans']    # (bsz, #queries, 2) center-width, normalized
    # Note: do NOT use pred_iou for scoring here - it may be randomly initialized
    # if resuming from a checkpoint without iou_embed training

    prob = F.softmax(pred_logits, dim=-1)
    scores = prob[..., 0]  # (bsz, #queries) foreground score only

    bsz, n_queries = scores.shape
    span_labels = targets["span_labels"]

    total_score_loss = torch.tensor(0.0, device=scores.device)
    total_span_loss = torch.tensor(0.0, device=scores.device)
    n_valid = 0

    for i in range(bsz):
        gt_cxw = span_labels[i]["spans"]
        if gt_cxw.shape[0] == 0:
            continue

        # IoU per query (detached for reward computation)
        with torch.no_grad():
            ious = _compute_query_ious(pred_spans[i].detach(), gt_cxw)  # (nq,)

            # Reward shaping
            if reward_mode == "step":
                # Binary step rewards aligned with eval thresholds
                rewards = 0.5 * (ious > 0.5).float() + 0.5 * (ious > 0.7).float()
            elif reward_mode == "iou_step":
                # Raw IoU + bonus at thresholds
                rewards = ious.clone() + 0.1 * (ious > 0.3).float() + \
                          0.2 * (ious > 0.5).float() + 0.3 * (ious > 0.7).float()
            else:
                # Raw IoU (original)
                rewards = ious.clone()

            if boundary_bonus or threshold_steps:
                # Compute per-query boundary distances for enhanced rewards
                pred_xx_r = span_cxw_to_xx(pred_spans[i].detach())  # (nq, 2)
                gt_xx_r = span_cxw_to_xx(gt_cxw)  # (ngt, 2)
                # Find best-matching GT per query
                ps = pred_xx_r[:, 0].unsqueeze(1)  # (nq, 1)
                pe = pred_xx_r[:, 1].unsqueeze(1)
                gs = gt_xx_r[:, 0].unsqueeze(0)    # (1, ngt)
                ge = gt_xx_r[:, 1].unsqueeze(0)
                inter_ = (torch.min(pe, ge) - torch.max(ps, gs)).clamp(min=0)
                union_ = (pe - ps).clamp(min=1e-6) + (ge - gs).clamp(min=1e-6) - inter_
                iou_mat_r = inter_ / union_.clamp(min=1e-6)  # (nq, ngt)
                best_gt_per_q = iou_mat_r.argmax(dim=1)  # (nq,)

            if boundary_bonus:
                # Bonus for boundary precision: Gaussian centered at 0 error
                matched_gt_xx = gt_xx_r[best_gt_per_q]  # (nq, 2)
                gt_dur = (matched_gt_xx[:, 1] - matched_gt_xx[:, 0]).clamp(min=1e-6)
                start_err = (pred_xx_r[:, 0] - matched_gt_xx[:, 0]).abs() / gt_dur
                end_err = (pred_xx_r[:, 1] - matched_gt_xx[:, 1]).abs() / gt_dur
                # Gaussian bonus: peaks at 0 error, sigma=0.2
                boundary_reward = 0.5 * (torch.exp(-start_err**2 / 0.08) +
                                         torch.exp(-end_err**2 / 0.08))
                rewards = rewards + 0.15 * boundary_reward

            if threshold_steps:
                # Stepping stone bonuses at IoU thresholds
                rewards = rewards + 0.05 * (ious > 0.3).float() + \
                          0.10 * (ious > 0.5).float() + \
                          0.15 * (ious > 0.7).float()

            # --- IC reward (VTG-Reasoner): Intersection Compactness ---
            # R_IC = |GT ∩ pred| / |pred|, only when IoU >= 0.5
            # Penalizes over-wide predictions that capture GT but include irrelevant frames.
            if ic_reward_weight > 0.0:
                pred_widths = pred_spans[i].detach()[:, 1].clamp(min=1e-6)  # center-width fmt
                avg_gt_width = gt_cxw[:, 1].mean().clamp(min=1e-6)
                # intersect = iou * (pred_w + gt_w) / (1 + iou)  [from union-intersection formula]
                intersect = ious * (pred_widths + avg_gt_width) / (1.0 + ious + 1e-6)
                ic = (intersect / pred_widths).clamp(0.0, 1.0)
                ic = ic * (ious >= 0.5).float()  # only active when IoU >= 0.5
                rewards = rewards + ic_reward_weight * ic

            # --- CenDist reward (LongVTG-R1): Center Distance ---
            # R_CenDist = 1 - |center_pred - center_gt| / Duration
            # Provides dense gradient even when IoU=0 (predictions nearby but non-overlapping).
            if cendist_weight > 0.0:
                pred_centers = pred_spans[i].detach()[:, 0]  # center coordinate, normalized [0,1]
                gt_center = gt_cxw[:, 0].mean()  # mean GT center (normalized)
                center_dist = (pred_centers - gt_center).abs().clamp(0.0, 1.0)
                cendist_reward = 1.0 - center_dist  # (nq,) in [0, 1]
                # Blend: base_reward * (1-w) + cendist * w  OR additive with weight
                rewards = (1.0 - cendist_weight) * rewards + cendist_weight * cendist_reward

        # --- Component 1: Score REINFORCE ---
        query_scores = scores[i]
        # Determine baseline based on mode (or legacy iou_greedy flag)
        _mode = "iou_greedy" if iou_greedy else baseline_mode

        if _mode == "grpo":
            # GRPO-style: vectorized advantage over ALL queries simultaneously.
            # advantage_i = (reward_i - mean_reward) / std_reward
            # Policy gradient applied to all queries at once, not just sampled ones.
            # This avoids the collapsed-baseline problem of score_greedy and is
            # equivalent to GRPO with group size = n_queries.
            group_mean = rewards.mean()
            group_std = rewards.std().clamp(min=1e-8)
            advantages = (rewards - group_mean) / group_std  # (nq,)
            if clip_adv > 0:
                advantages = advantages.clamp(-clip_adv, clip_adv)
            log_probs_all = F.log_softmax(query_scores / temperature, dim=0)  # (nq,)
            score_loss_i = -(advantages * log_probs_all).sum()
        else:
            if _mode == "iou_greedy":
                # Option-B: IoU-best query as baseline (task-metric aligned)
                greedy_reward = rewards.max()
            elif _mode == "loo_mean":
                # Option-C: mean of all query rewards (leave-one-out estimator, lowest variance)
                greedy_reward = rewards.mean()
            else:
                # Original (score_greedy): score-best query as baseline
                greedy_reward = rewards[query_scores.argmax()]

            # Multi-sample REINFORCE for lower variance
            sample_logits = query_scores / temperature
            sample_probs = F.softmax(sample_logits, dim=0)
            dist = torch.distributions.Categorical(sample_probs)

            score_loss_i = torch.tensor(0.0, device=scores.device)
            for _ in range(n_samples):
                sample_idx = dist.sample()
                sample_log_prob = dist.log_prob(sample_idx)
                advantage = rewards[sample_idx] - greedy_reward
                score_loss_i = score_loss_i - advantage * sample_log_prob
            score_loss_i = score_loss_i / n_samples

        # --- Component 2: Reward-weighted span regression ---
        # Only target top-k foreground queries (those the model is most confident about)
        # to avoid conflicting with supervised loss on background queries.
        fg_scores = prob[i, :, 0].detach()  # (nq,)
        _top_k = min(top_k, n_queries)  # configurable, default 3
        _, top_indices = fg_scores.topk(_top_k)

        gt_xx = span_cxw_to_xx(gt_cxw)
        top_pred_cxw = pred_spans[i][top_indices]  # (k, 2) WITH gradient
        top_ious = ious[top_indices]  # (k,)

        # Find best-matching GT per top query
        top_pred_xx = span_cxw_to_xx(top_pred_cxw)
        pred_s = top_pred_xx[:, 0].unsqueeze(1)
        pred_e = top_pred_xx[:, 1].unsqueeze(1)
        gt_s = gt_xx[:, 0].unsqueeze(0)
        gt_e = gt_xx[:, 1].unsqueeze(0)
        inter_s = torch.max(pred_s, gt_s)
        inter_e = torch.min(pred_e, gt_e)
        inter = (inter_e - inter_s).clamp(min=0)
        pred_len = (pred_e - pred_s).clamp(min=1e-6)
        gt_len = (gt_e - gt_s).clamp(min=1e-6)
        union = pred_len + gt_len - inter
        iou_mat = inter / union.clamp(min=1e-6)  # (k, ngt)
        best_gt_idx = iou_mat.detach().argmax(dim=1)  # (k,)
        best_gt_cxw = gt_cxw[best_gt_idx]  # (k, 2)

        # Weight by span_weight_mode
        if span_weight_mode == "balanced":
            # Peak at IoU=0.5: focus on "half-right" predictions with clear refinement direction
            raw_weights = top_ious * (1.0 - top_ious)  # (k,)
        elif span_weight_mode == "iou":
            # Focus on already-good predictions
            raw_weights = top_ious  # (k,)
        else:  # "improvement"
            raw_weights = (1.0 - top_ious).clamp(min=0)  # (k,)
        if threshold_aware:
            threshold_bonus = torch.exp(-((top_ious - 0.7) ** 2) / (2 * 0.15 ** 2))
            weights = raw_weights * (1.0 + threshold_bonus)
        else:
            weights = raw_weights
        weights = weights / (weights.sum() + 1e-8)

        # Weighted L1 on center-width
        span_diff = (top_pred_cxw - best_gt_cxw).abs()  # (k, 2) differentiable
        span_loss_i = (weights.unsqueeze(1) * span_diff).sum()

        # Optionally add GIoU loss for better boundary gradients
        if use_giou:
            top_pred_xx_giou = span_cxw_to_xx(top_pred_cxw)  # (k, 2) with gradient
            best_gt_xx = span_cxw_to_xx(best_gt_cxw)  # (k, 2)
            giou_vals = generalized_temporal_iou(top_pred_xx_giou, best_gt_xx).diag()  # (k,)
            giou_loss_i = (weights * (1 - giou_vals)).sum()
            span_loss_i = span_loss_i + giou_loss_i

        total_score_loss = total_score_loss + score_loss_i
        total_span_loss = total_span_loss + span_loss_i
        n_valid += 1

    if n_valid > 0:
        total_score_loss = total_score_loss / n_valid
        total_span_loss = total_span_loss / n_valid

    if decompose:
        return total_score_loss, total_span_loss
    return total_score_loss + total_span_loss


def top1_refinement_loss(outputs, targets):
    """Top-1 query targeted refinement loss.

    For each sample, find the highest-scoring query (by foreground prob),
    then compute L1+GIoU loss between its span and the closest GT span.
    This directly targets the mismatch between Hungarian matching (training)
    and top-1 selection (evaluation).
    """
    pred_logits = outputs['pred_logits']  # (bsz, #queries, 2)
    pred_spans = outputs['pred_spans']    # (bsz, #queries, 2) center-width
    prob = F.softmax(pred_logits, dim=-1)
    fg_scores = prob[..., 0]  # (bsz, #queries) foreground scores

    bsz = fg_scores.shape[0]
    span_labels = targets["span_labels"]

    total_l1 = torch.tensor(0.0, device=fg_scores.device)
    total_giou = torch.tensor(0.0, device=fg_scores.device)
    n_valid = 0

    for i in range(bsz):
        gt_cxw = span_labels[i]["spans"]
        if gt_cxw.shape[0] == 0:
            continue

        # Find top-1 query by foreground score
        top1_idx = fg_scores[i].argmax()
        top1_span = pred_spans[i][top1_idx].unsqueeze(0)  # (1, 2) WITH gradient

        # Find closest GT to this query's span
        with torch.no_grad():
            top1_xx = span_cxw_to_xx(top1_span)
            gt_xx = span_cxw_to_xx(gt_cxw)
            p_s, p_e = top1_xx[0, 0], top1_xx[0, 1]
            g_s, g_e = gt_xx[:, 0], gt_xx[:, 1]
            inter = (torch.min(p_e, g_e) - torch.max(p_s, g_s)).clamp(min=0)
            union = (p_e - p_s).clamp(min=1e-6) + (g_e - g_s).clamp(min=1e-6) - inter
            iou = inter / union.clamp(min=1e-6)
            best_gt_idx = iou.argmax()

        best_gt_cxw = gt_cxw[best_gt_idx].unsqueeze(0)  # (1, 2)

        # L1 loss on center-width
        l1_loss = F.l1_loss(top1_span, best_gt_cxw, reduction='mean')

        # GIoU loss
        top1_xx_grad = span_cxw_to_xx(top1_span)
        best_gt_xx = span_cxw_to_xx(best_gt_cxw)
        giou_loss = 1 - generalized_temporal_iou(top1_xx_grad, best_gt_xx).diag().mean()

        total_l1 = total_l1 + l1_loss
        total_giou = total_giou + giou_loss
        n_valid += 1

    if n_valid > 0:
        total_l1 = total_l1 / n_valid
        total_giou = total_giou / n_valid

    return total_l1 + total_giou


def _compute_curriculum_coefs(opt, epoch_i):
    """Compute effective rl coefficients based on progressive RL schedule.

    Returns:
        (score_reinforce_coef, span_regression_coef): effective weights for the two SCST components.
        Their sum equals the effective rl_coef for this epoch.
    """
    rl_coef = getattr(opt, 'rl_coef', 0.1)
    curriculum = getattr(opt, 'progressive_rl', 'none')

    if curriculum == 'linear':
        # Linear warmup: rl_coef goes from 0 to target over warmup epochs
        warmup = getattr(opt, 'rl_warmup_epochs', 0)
        if warmup > 0 and epoch_i < warmup:
            progress = (epoch_i + 1) / warmup  # 1/N ... N/N
            effective_coef = rl_coef * progress
        else:
            effective_coef = rl_coef
        return effective_coef, effective_coef  # both components scale together

    elif curriculum == 'three_phase':
        # Phase 1 [0, phase1): top1_refine only, no RL at all
        # Phase 2 [phase1, phase1+phase2): reward-weighted span regression only (gentle)
        # Phase 3 [phase1+phase2, ...): full RL (score REINFORCE + span regression)
        phase1 = getattr(opt, 'phase1_epochs', 15)
        phase2 = getattr(opt, 'phase2_epochs', 25)
        phase2_start = phase1
        phase3_start = phase1 + phase2

        if epoch_i < phase1:
            # Phase 1: no RL
            return 0.0, 0.0
        elif epoch_i < phase3_start:
            # Phase 2: only span regression, linearly ramp up
            progress = (epoch_i - phase2_start + 1) / phase2
            span_coef = rl_coef * progress
            return 0.0, span_coef  # score_reinforce=0, span_regression=ramping
        else:
            # Phase 3: full RL or span-only depending on phase3_span_only flag
            phase3_span_only = getattr(opt, 'phase3_span_only', False)
            base_span = rl_coef

            if phase3_span_only:
                # Phase 3 = enhanced span regression only (no score REINFORCE)
                # Allows using better reward shaping / top_k without policy gradient noise
                base_reinforce = 0.0
            else:
                # Standard Phase 3: linearly ramp in score REINFORCE over 10 epochs
                ramp_epochs = 10
                reinforce_progress = min(1.0, (epoch_i - phase3_start + 1) / ramp_epochs)
                base_reinforce = rl_coef * reinforce_progress

            # Optionally decay rl_coef in Phase3 to prevent degradation
            phase3_decay = getattr(opt, 'rl_phase3_decay', 'none')
            if phase3_decay == 'cosine':
                import math
                n_epoch = getattr(opt, 'n_epoch', 80)
                phase3_len = max(1, n_epoch - phase3_start)
                phase3_progress = (epoch_i - phase3_start) / phase3_len
                # Cosine decay from 1.0 to 0.2
                decay_factor = 0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * phase3_progress))
                base_reinforce *= decay_factor
                base_span *= decay_factor

            return base_reinforce, base_span

    else:
        # No curriculum: fixed rl_coef for both
        return rl_coef, rl_coef


def train_epoch_scst(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer, ema=None):
    """Training epoch with SCST RL / top-1 refinement, optionally combined with supervised loss."""
    logger.info(f"[SCST Epoch {epoch_i+1}]")
    model.train()
    criterion.train()

    time_meters = defaultdict(AverageMeter)
    loss_meters = defaultdict(AverageMeter)

    # Curriculum: compute effective RL coefficients for this epoch
    reinforce_coef, span_reg_coef = _compute_curriculum_coefs(opt, epoch_i)
    use_decomposed = (reinforce_coef != span_reg_coef)
    rl_coef_total = getattr(opt, 'rl_coef', 0.1)  # for logging only

    rl_only = getattr(opt, 'rl_only', False)
    use_top1 = getattr(opt, 'top1_refine', False)
    top1_coef = getattr(opt, 'top1_coef', 1.0)
    threshold_aware = getattr(opt, 'threshold_aware_reward', False)
    span_weight_mode = getattr(opt, 'span_weight_mode', 'improvement')
    use_giou = getattr(opt, 'scst_giou', False)
    n_samples = getattr(opt, 'rl_n_samples', 3)
    boundary_bonus = getattr(opt, 'scst_boundary_bonus', False)
    threshold_steps = getattr(opt, 'scst_threshold_steps', False)
    iou_greedy = getattr(opt, 'scst_iou_greedy', False)
    baseline_mode = getattr(opt, 'scst_baseline_mode', 'score_greedy')
    reward_mode = getattr(opt, 'scst_reward_mode', 'iou')
    clip_adv = getattr(opt, 'scst_clip_adv', 0.0)
    ic_reward_weight = getattr(opt, 'scst_ic_reward_weight', 0.0)
    cendist_weight = getattr(opt, 'scst_cendist_weight', 0.0)
    phase3_anchor_weight = getattr(opt, 'phase3_anchor_weight', 0.0)
    scst_top_k = getattr(opt, 'scst_top_k', 3)

    # Determine Phase3 status for Option-A anchor
    _curriculum = getattr(opt, 'progressive_rl', 'none')
    _in_phase3 = False
    if _curriculum == 'three_phase' and phase3_anchor_weight > 0.0:
        _p1 = getattr(opt, 'phase1_epochs', 15)
        _p2 = getattr(opt, 'phase2_epochs', 25)
        _in_phase3 = (epoch_i >= _p1 + _p2)

    curriculum = getattr(opt, 'progressive_rl', 'none')
    if curriculum != 'none':
        logger.info(f"[Curriculum] epoch={epoch_i+1}, reinforce_coef={reinforce_coef:.4f}, "
                     f"span_reg_coef={span_reg_coef:.4f}")

    num_training_examples = len(train_loader)
    timer_dataloading = time.time()
    for batch_idx, batch in tqdm(enumerate(train_loader),
                                 desc="SCST Training",
                                 total=num_training_examples):
        time_meters["dataloading_time"].update(time.time() - timer_dataloading)

        timer_start = time.time()
        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)
        time_meters["prepare_inputs_time"].update(time.time() - timer_start)

        timer_start = time.time()
        outputs = model(**model_inputs)

        # Supervised loss
        loss_dict = criterion(outputs, targets, epoch_i)
        weight_dict = criterion.weight_dict
        if rl_only:
            sup_loss = torch.tensor(0.0, device=opt.device)
            # Still train IoU head even in rl_only mode (it has its own parameters)
            if 'loss_iou' in loss_dict and 'loss_iou' in weight_dict:
                sup_loss = loss_dict['loss_iou'] * weight_dict['loss_iou']
            # Option-A: Phase3 supervised anchor to prevent RL drift
            if _in_phase3 and phase3_anchor_weight > 0.0:
                full_sup = sum(loss_dict[k] * weight_dict[k]
                               for k in loss_dict.keys() if k in weight_dict)
                sup_loss = sup_loss + phase3_anchor_weight * full_sup
        else:
            sup_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # SCST RL loss — curriculum-aware
        if use_decomposed:
            score_rl, span_rl = scst_rl_loss(outputs, targets,
                                              span_loss_type=opt.span_loss_type,
                                              n_samples=n_samples,
                                              decompose=True,
                                              threshold_aware=threshold_aware,
                                              span_weight_mode=span_weight_mode,
                                              use_giou=use_giou,
                                              boundary_bonus=boundary_bonus,
                                              threshold_steps=threshold_steps,
                                              iou_greedy=iou_greedy,
                                              baseline_mode=baseline_mode,
                                              reward_mode=reward_mode,
                                              clip_adv=clip_adv,
                                              ic_reward_weight=ic_reward_weight,
                                              cendist_weight=cendist_weight,
                                              top_k=scst_top_k)
            rl_loss_combined = reinforce_coef * score_rl + span_reg_coef * span_rl
            rl_loss_scalar = float(score_rl + span_rl)  # for logging
        else:
            rl_loss = scst_rl_loss(outputs, targets, span_loss_type=opt.span_loss_type,
                                   n_samples=n_samples,
                                   threshold_aware=threshold_aware,
                                   span_weight_mode=span_weight_mode,
                                   use_giou=use_giou,
                                   boundary_bonus=boundary_bonus,
                                   threshold_steps=threshold_steps,
                                   iou_greedy=iou_greedy,
                                   baseline_mode=baseline_mode,
                                   reward_mode=reward_mode,
                                   clip_adv=clip_adv,
                                   ic_reward_weight=ic_reward_weight,
                                   cendist_weight=cendist_weight,
                                   top_k=scst_top_k)
            rl_loss_combined = reinforce_coef * rl_loss
            rl_loss_scalar = float(rl_loss)

        # Top-1 refinement loss
        t1_loss = torch.tensor(0.0, device=opt.device)
        if use_top1:
            t1_loss = top1_refinement_loss(outputs, targets)

        # Score-IoU pairwise ranking loss
        rank_loss = torch.tensor(0.0, device=opt.device)
        rank_coef = getattr(opt, 'score_iou_rank_coef', 0.0)
        if rank_coef > 0:
            rank_margin = getattr(opt, 'score_iou_rank_margin', 0.1)
            rank_iou_threshold = getattr(opt, 'score_iou_rank_iou_threshold', 0.1)
            rank_loss = score_iou_ranking_loss(outputs, targets,
                                               margin=rank_margin,
                                               iou_threshold=rank_iou_threshold)

        # Combined loss
        losses = sup_loss + rl_loss_combined + top1_coef * t1_loss + rank_coef * rank_loss
        time_meters["model_forward_time"].update(time.time() - timer_start)

        timer_start = time.time()
        optimizer.zero_grad()
        losses.backward()
        # Log gradient diagnostics for first batch of first epoch
        if batch_idx == 0 and epoch_i == 0:
            span_grad = sum(p.grad.norm().item() for p in model.span_embed.parameters() if p.grad is not None)
            class_grad = sum(p.grad.norm().item() for p in model.class_embed.parameters() if p.grad is not None)
            iou_grad = sum(p.grad.norm().item() for p in model.iou_embed.parameters() if p.grad is not None)
            logger.info(f"[SCST Grad Check] span_embed: {span_grad:.6f}, class_embed: {class_grad:.6f}, iou_embed: {iou_grad:.6f}")
        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        time_meters["model_backward_time"].update(time.time() - timer_start)

        loss_dict["loss_overall"] = float(losses)
        loss_dict["loss_rl"] = rl_loss_scalar
        loss_dict["loss_top1"] = float(t1_loss)
        if use_decomposed:
            loss_dict["loss_rl_reinforce"] = float(score_rl)
            loss_dict["loss_rl_span_reg"] = float(span_rl)
        for k, v in loss_dict.items():
            loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

        timer_dataloading = time.time()
        if opt.debug and batch_idx == 3:
            break

    tb_writer.add_scalar("Train/lr", float(optimizer.param_groups[0]["lr"]), epoch_i+1)
    for k, v in loss_meters.items():
        tb_writer.add_scalar("Train/{}".format(k), v.avg, epoch_i+1)

    to_write = opt.train_log_txt_formatter.format(
        time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
        epoch=epoch_i+1,
        loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()]))
    with open(opt.train_log_filepath, "a") as f:
        f.write(to_write)

    logger.info("Epoch time stats:")
    for name, meter in time_meters.items():
        d = {k: f"{getattr(meter, k):.4f}" for k in ["max", "min", "avg"]}
        logger.info(f"{name} ==> {d}")


def calculate_stop_score(metrics: dict[str, float], opts):
    # Define the weights
    if opts.dset_name in ['hl']:
        weights = {
            "MR-full-R1@0.3": 0.3,
            "MR-full-R1@0.5": 0.5,
            "MR-full-R1@0.7": 0.7,
            "MR-full-mAP@0.5": 0.5,
            "MR-full-mAP@0.75": 0.75,
            "MR-full-mAP": 0.25,
            "MR-long-mAP": 0.25,
            "MR-middle-mAP": 0.25,
            "MR-short-mAP": 0.25,
        }
    else:
        weights = {
            "MR-full-R1@0.3": 0.3,
            "MR-full-R1@0.5": 0.5,
            "MR-full-R1@0.7": 0.7,
            "MR-full-mIoU": 0.50,
        }

    # Normalize weights
    total_weight = sum(weights.values())
    normalized_weights = {key: value / total_weight for key, value in weights.items()}

    # Compute the score
    score = 0.0
    for metric, weight in normalized_weights.items():
        if metric in metrics:
            score += metrics[metric] * weight

    return score

def get_best_score(metrics: dict[str, float], opt):
    if getattr(opt, "best_metric_strategy", "weighted_stop_score") == "main_metric":
        return metrics[opt.main_metric]
    return calculate_stop_score(metrics, opts=opt)


def train(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt):
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)

    # EMA setup
    ema_decay = getattr(opt, 'ema_decay', 0.0)
    use_ema = ema_decay > 0
    ema = None
    if use_ema:
        ema = ModelEMA(model, decay=ema_decay)
        logger.info(f"[EMA] Enabled with decay={ema_decay}")

    tb_writer = SummaryWriter(opt.tensorboard_log_dir)
    tb_writer.add_text("hyperparameters", dict_to_markdown(vars(opt), max_str_len=None))
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"

    # setup_model already applies the freeze policy before optimizer creation.
    if getattr(opt, 'freeze_backbone', False):
        head_prefixes, extra_prefixes, _ = build_finetune_prefixes(opt)
        logger.info(
            f"[Freeze Backbone] Confirmed active in train(). Heads: {', '.join(sorted(head_prefixes))}; "
            f"extra: {', '.join(sorted(extra_prefixes)) if extra_prefixes else 'none'}"
        )

    # if opt.a_feat_dir is None:
    train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
        pin_memory=opt.pin_memory
    )
    # else:
    #     train_loader = DataLoader(
    #         train_dataset,
    #         collate_fn=start_end_collate_audio,
    #         batch_size=opt.bsz,
    #         num_workers=opt.num_workers,
    #         shuffle=True,
    #         pin_memory=opt.pin_memory
    #     )

    prev_best_score = 0.
    es_cnt = 0
    # start_epoch = 0
    if opt.start_epoch is None:
        start_epoch = -1 if opt.eval_untrained else 0
    else:
        start_epoch = opt.start_epoch
    save_submission_filename = "latest_{}_{}_preds.jsonl".format(opt.dset_name, opt.eval_split_name)
    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        if epoch_i > -1:
            if getattr(opt, 'scst', False):
                train_epoch_scst(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer, ema=ema)
            else:
                train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer, ema=ema)
            lr_scheduler.step()
        eval_epoch_interval = 5
        if opt.eval_path is not None and (epoch_i + 1) % eval_epoch_interval == 0:
            eval_model = ema.shadow if use_ema else model
            with torch.no_grad():
                metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = \
                    eval_epoch(eval_model, val_dataset, opt, save_submission_filename, epoch_i, criterion, tb_writer)

            # log
            to_write = opt.eval_log_txt_formatter.format(
                time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                epoch=epoch_i,
                loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in eval_loss_meters.items()]),
                eval_metrics_str=json.dumps(metrics_no_nms))

            with open(opt.eval_log_filepath, "a") as f:
                f.write(to_write)
            logger.info("metrics_no_nms {}".format(pprint.pformat(metrics_no_nms["brief"], indent=4)))
            if metrics_nms is not None:
                logger.info("metrics_nms {}".format(pprint.pformat(metrics_nms["brief"], indent=4)))

            metrics = metrics_no_nms
            for k, v in metrics["brief"].items():
                tb_writer.add_scalar(f"Eval/{k}", float(v), epoch_i+1)

            stop_score = get_best_score(metrics["brief"], opt)

            if stop_score > prev_best_score:
                es_cnt = 0
                prev_best_score = stop_score

                checkpoint = {
                    "model": ema.state_dict() if use_ema else model.state_dict(),
                    "model_raw": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch_i,
                    "opt": opt
                }
                torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"))

                best_file_paths = [e.replace("latest", "best") for e in latest_file_paths]
                for src, tgt in zip(latest_file_paths, best_file_paths):
                    os.renames(src, tgt)
                logger.info(
                    f"The checkpoint file has been updated with score {stop_score} "
                    f"(strategy={opt.best_metric_strategy})"
                )
            else:
                es_cnt += 1
                if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt:  # early stop
                    with open(opt.train_log_filepath, "a") as f:
                        f.write(f"Early Stop at epoch {epoch_i}")
                    logger.info(f"\n>>>>> Early stop at epoch {epoch_i}  {prev_best_score}\n")
                    break

            # save ckpt
            checkpoint = {
                "model": ema.state_dict() if use_ema else model.state_dict(),
                "model_raw": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch_i,
                "opt": opt
            }
            torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_latest.ckpt"))

        save_interval = 10 if "subs_train" in opt.train_path else 50  # smaller for pretrain
        if (epoch_i + 1) % save_interval == 0 or (epoch_i + 1) % opt.lr_drop == 0:  # additional copies
            checkpoint = {
                "model": ema.state_dict() if use_ema else model.state_dict(),
                "model_raw": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch_i,
                "opt": opt
            }
            torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", f"_e{epoch_i:04d}.ckpt"))

        if opt.debug:
            break

    tb_writer.close()



def train_hl(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt):
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)

    tb_writer = SummaryWriter(opt.tensorboard_log_dir)
    tb_writer.add_text("hyperparameters", dict_to_markdown(vars(opt), max_str_len=None))
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"

    train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
        pin_memory=opt.pin_memory
    )

    prev_best_score = 0.
    es_cnt = 0
    # start_epoch = 0
    if opt.start_epoch is None:
        start_epoch = -1 if opt.eval_untrained else 0
    else:
        start_epoch = opt.start_epoch
    save_submission_filename = "latest_{}_{}_preds.jsonl".format(opt.dset_name, opt.eval_split_name)
    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        if epoch_i > -1:
            train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer)
            lr_scheduler.step()
        eval_epoch_interval = 5
        if opt.eval_path is not None and (epoch_i + 1) % eval_epoch_interval == 0:
            with torch.no_grad():
                metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = \
                    eval_epoch(model, val_dataset, opt, save_submission_filename, epoch_i, criterion, tb_writer)

            # log
            to_write = opt.eval_log_txt_formatter.format(
                time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                epoch=epoch_i,
                loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in eval_loss_meters.items()]),
                eval_metrics_str=json.dumps(metrics_no_nms))

            with open(opt.eval_log_filepath, "a") as f:
                f.write(to_write)
            logger.info("metrics_no_nms {}".format(pprint.pformat(metrics_no_nms["brief"], indent=4)))
            if metrics_nms is not None:
                logger.info("metrics_nms {}".format(pprint.pformat(metrics_nms["brief"], indent=4)))

            metrics = metrics_no_nms
            for k, v in metrics["brief"].items():
                tb_writer.add_scalar(f"Eval/{k}", float(v), epoch_i+1)

            # stop_score = metrics["brief"]["MR-full-mAP"]
            stop_score = metrics["brief"]["mAP"]
            if stop_score > prev_best_score:
                es_cnt = 0
                prev_best_score = stop_score

                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch_i,
                    "opt": opt
                }
                torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"))

                best_file_paths = [e.replace("latest", "best") for e in latest_file_paths]
                for src, tgt in zip(latest_file_paths, best_file_paths):
                    os.renames(src, tgt)
                logger.info("The checkpoint file has been updated.")
            else:
                es_cnt += 1
                if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt:  # early stop
                    with open(opt.train_log_filepath, "a") as f:
                        f.write(f"Early Stop at epoch {epoch_i}")
                    logger.info(f"\n>>>>> Early stop at epoch {epoch_i}  {prev_best_score}\n")
                    break

            # save ckpt
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch_i,
                "opt": opt
            }
            # torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_latest.ckpt"))

        save_interval = 10 if "subs_train" in opt.train_path else 50  # smaller for pretrain
        if (epoch_i + 1) % save_interval == 0 or (epoch_i + 1) % opt.lr_drop == 0:  # additional copies
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch_i,
                "opt": opt
            }
            # torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", f"_e{epoch_i:04d}.ckpt"))

        if opt.debug:
            break

    tb_writer.close()




def start_training():
    logger.info("Setup config, data and model...")
    opt = BaseOptions().parse()
    set_seed(opt.seed)
    if opt.debug:  # keep the model run deterministically
        # 'cudnn.benchmark = True' enabled auto finding the best algorithm for a specific input/net config.
        # Enable this only when input size is fixed.
        cudnn.benchmark = False
        cudnn.deterministic = True
    print('##################')
    print(opt.a_feat_dir is None)
    print(opt.a_feat_dir)
    print('##################')
    dataset_config = dict(
        dset_name=opt.dset_name,
        data_path=opt.train_path,
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dirs=opt.t_feat_dirs,
        a_feat_dirs=opt.a_feat_dirs if opt.a_feat_dim > 0 else None,
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
        txt_drop_ratio=opt.txt_drop_ratio,
        dset_domain=opt.dset_domain,
    )

    dataset_config["data_path"] = opt.train_path
    train_dataset = StartEndDataset(**dataset_config)


    if opt.eval_path is not None:
        dataset_config["data_path"] = opt.eval_path
        dataset_config["txt_drop_ratio"] = 0
        # dataset_config["q_feat_dirs"] = opt.t_feat_dir.replace("sub_features", "text_features")  # for pretraining
        # dataset_config["load_labels"] = False  # uncomment to calculate eval loss
        # if opt.a_feat_dir is None:
        eval_dataset = StartEndDataset(**dataset_config)
        # else:
        #     eval_dataset = StartEndDataset_audio(**dataset_config)
    else:
        eval_dataset = None

    model, criterion, optimizer, lr_scheduler = setup_model(opt)
    logger.info(f"Model {model}")
    count_parameters(model)
    logger.info("Start Training...")
    
    # For tvsum dataset, use train_hl function
    if opt.dset_name in ['tvsum']:
        train_hl(model, criterion, optimizer, lr_scheduler, train_dataset, eval_dataset, opt)
    else:
        train(model, criterion, optimizer, lr_scheduler, train_dataset, eval_dataset, opt)
    
    return opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"), opt.eval_split_name, opt.eval_path, opt.debug, opt


if __name__ == '__main__':
    best_ckpt_path, eval_split_name, eval_path, debug, opt = start_training()
    if not debug:
        input_args = ["--resume", best_ckpt_path,
                      "--eval_split_name", eval_split_name,
                      "--eval_path", eval_path]

        import sys
        sys.argv[1:] = input_args
        logger.info("\n\n\nFINISHED TRAINING!!!")
        logger.info("Evaluating model at {}".format(best_ckpt_path))
        logger.info("Input args {}".format(sys.argv[1:]))
        start_inference(opt)
