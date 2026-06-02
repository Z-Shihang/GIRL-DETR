"""
RL Refinement Agent for GIRL Moment Retrieval.

Two-phase training:
  Phase 1 (Supervised):  Train policy with oracle (GT-directed) actions.
  Phase 2 (RL):          Fine-tune with REINFORCE / PPO using IoU reward.

Architecture:
  - Discrete: Categorical policy on separate start/end boundary adjustments
    with optional GRU memory, gate attention, and IoU auxiliary prediction
    (inspired by TSP-PRL, Wu et al., AAAI 2020)
  - Continuous: Gaussian policy on (delta_center, delta_width)

Reference papers:
  - Read, Watch, and Move (He et al., AAAI 2019) arXiv:1901.06829
  - TSP-PRL (Wu et al., AAAI 2020) arXiv:2001.06680
  - BAR (Wu et al., ACM MM 2020) arXiv:2009.08614
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np

from girl.span_utils import span_cxw_to_xx, generalized_temporal_iou


# =============================================================================
# Continuous Policy (original)
# =============================================================================

class SpanRefinementPolicy(nn.Module):
    """
    Actor-Critic policy network for span boundary refinement (continuous actions).
    Action: continuous (delta_center, delta_width) adjustments.
    """

    def __init__(self, hidden_dim=256, state_dim=None, max_action=0.1):
        super().__init__()
        if state_dim is None:
            state_dim = 2 + 1 + hidden_dim + hidden_dim
        self.max_action = max_action

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.actor_mean = nn.Linear(hidden_dim, 2)
        self.actor_log_std = nn.Parameter(torch.zeros(2) - 3.0)

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.actor_mean.weight, -0.01, 0.01)
        nn.init.zeros_(self.actor_mean.bias)

    def forward(self, state):
        features = self.shared(state)
        action_mean = torch.tanh(self.actor_mean(features)) * self.max_action
        value = self.critic(features)
        return action_mean, self.actor_log_std, value

    def get_action_and_value(self, state, action=None, deterministic=False):
        action_mean, action_log_std, value = self(state)
        action_std = action_log_std.exp().expand_as(action_mean)
        dist = Normal(action_mean, action_std)

        if action is None:
            if deterministic:
                action = action_mean.clamp(-self.max_action, self.max_action)
            else:
                action = dist.sample()
                action = action.clamp(-self.max_action, self.max_action)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy, value.squeeze(-1)

    def get_value(self, state):
        features = self.shared(state)
        return self.critic(features).squeeze(-1)


# =============================================================================
# Discrete Policy (TSP-PRL inspired)
# =============================================================================

class DiscreteSpanPolicy(nn.Module):
    """
    Discrete action space policy for span boundary refinement.
    Inspired by TSP-PRL (Wu et al., AAAI 2020).

    Key design: **Coarse-to-fine multi-scale actions** across refinement steps.
      Step 0: large adjustments  [-8, -4, -2, 0, 2, 4, 8] clips
      Step 1: medium adjustments [-4, -2, -1, 0, 1, 2, 4] clips
      Step 2: fine adjustments   [-2, -1, -0.5, 0, 0.5, 1, 2] clips (sub-clip)
    This mirrors TSP-PRL's tree-structured coarse-to-fine search.

    Separate action heads for start and end boundary adjustments.
    Optional: GRU step memory, gate attention, IoU auxiliary prediction.
    """

    NUM_ACTIONS = 7  # per boundary per step

    # Multi-scale action deltas (in clip units) for each refinement step
    MULTISCALE_DELTAS = [
        [-8, -4, -2, 0, 2, 4, 8],       # Step 0: coarse
        [-4, -2, -1, 0, 1, 2, 4],       # Step 1: medium
        [-2, -1, -0.5, 0, 0.5, 1, 2],   # Step 2: fine (sub-clip)
    ]

    def __init__(self, hidden_dim=256, state_dim=None, max_v_l=75,
                 use_gate_attn=False, use_gru=False, use_iou_aux=False,
                 num_refine_steps=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_v_l = max_v_l
        self.unit = 1.0 / max_v_l
        self.use_gate_attn = use_gate_attn
        self.use_gru = use_gru
        self.use_iou_aux = use_iou_aux
        self.num_refine_steps = num_refine_steps

        if state_dim is None:
            state_dim = 2 + 1 + hidden_dim * 4

        if use_gate_attn:
            self.gate_proj = nn.Linear(hidden_dim, hidden_dim)

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        if use_gru:
            self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Per-step action heads for coarse-to-fine refinement
        self.start_heads = nn.ModuleList([
            nn.Linear(hidden_dim, self.NUM_ACTIONS) for _ in range(num_refine_steps)
        ])
        self.end_heads = nn.ModuleList([
            nn.Linear(hidden_dim, self.NUM_ACTIONS) for _ in range(num_refine_steps)
        ])

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        if use_iou_aux:
            self.iou_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )

        # Register multi-scale delta buffers
        for step_i in range(num_refine_steps):
            delta_idx = min(step_i, len(self.MULTISCALE_DELTAS) - 1)
            self.register_buffer(
                f'boundary_deltas_{step_i}',
                torch.tensor(self.MULTISCALE_DELTAS[delta_idx], dtype=torch.float32)
            )
        # Keep a default for backward compatibility
        self.register_buffer(
            'boundary_deltas',
            torch.tensor(self.MULTISCALE_DELTAS[-1], dtype=torch.float32)
        )

        self._init_weights()

    def _get_deltas(self, step):
        """Get the boundary deltas tensor for a given refinement step."""
        return getattr(self, f'boundary_deltas_{step}')

    def _init_weights(self):
        no_change_idx = self.NUM_ACTIONS // 2  # index 3
        for step_i in range(self.num_refine_steps):
            nn.init.xavier_uniform_(self.start_heads[step_i].weight, gain=0.01)
            nn.init.zeros_(self.start_heads[step_i].bias)
            nn.init.xavier_uniform_(self.end_heads[step_i].weight, gain=0.01)
            nn.init.zeros_(self.end_heads[step_i].bias)
            # Small bias toward "no change" for stable initial exploration
            self.start_heads[step_i].bias.data[no_change_idx] = 0.5
            self.end_heads[step_i].bias.data[no_change_idx] = 0.5

    def forward(self, state, hidden=None, step=0):
        features = self.shared(state)
        if self.use_gru and hidden is not None:
            features = self.gru(features, hidden)
        new_hidden = features if self.use_gru else None

        step_idx = min(step, self.num_refine_steps - 1)
        start_logits = self.start_heads[step_idx](features)
        end_logits = self.end_heads[step_idx](features)
        value = self.critic(features).squeeze(-1)

        iou_pred = None
        if self.use_iou_aux:
            iou_pred = self.iou_head(features).squeeze(-1)

        return start_logits, end_logits, value, new_hidden, iou_pred

    def get_action_and_value(self, state, action=None, hidden=None, deterministic=False, step=0):
        start_logits, end_logits, value, new_hidden, iou_pred = self(state, hidden, step=step)

        start_dist = Categorical(logits=start_logits)
        end_dist = Categorical(logits=end_logits)

        if action is None:
            if deterministic:
                start_action = start_logits.argmax(dim=-1)
                end_action = end_logits.argmax(dim=-1)
            else:
                start_action = start_dist.sample()
                end_action = end_dist.sample()
            action = torch.stack([start_action, end_action], dim=-1)
        else:
            start_action = action[:, 0].long()
            end_action = action[:, 1].long()

        log_prob = start_dist.log_prob(start_action) + end_dist.log_prob(end_action)
        entropy = start_dist.entropy() + end_dist.entropy()

        return action, log_prob, entropy, value, new_hidden, iou_pred

    def action_to_delta(self, action, step=0):
        """Convert discrete action indices to (delta_center, delta_width) in normalized coords.

        Uses the multi-scale deltas corresponding to the given refinement step.
        """
        deltas = self._get_deltas(step)
        start_delta = deltas[action[:, 0].long()] * self.unit
        end_delta = deltas[action[:, 1].long()] * self.unit
        delta_center = (start_delta + end_delta) / 2
        delta_width = end_delta - start_delta
        return torch.stack([delta_center, delta_width], dim=-1)

    def get_value(self, state, hidden=None):
        features = self.shared(state)
        if self.use_gru and hidden is not None:
            features = self.gru(features, hidden)
        return self.critic(features).squeeze(-1)

    def compute_oracle_actions(self, pred_spans_cxw, gt_spans_cxw, step=0):
        """Compute the oracle (best) discrete action for each sample at a given step.

        Uses the multi-scale deltas for the specified step.

        Args:
            pred_spans_cxw: (N, 2) predicted (center, width)
            gt_spans_cxw: (N, 2) ground-truth (center, width)
            step: refinement step index (determines action granularity)

        Returns:
            oracle_start: (N,) best start action indices
            oracle_end: (N,) best end action indices
        """
        pred_xx = span_cxw_to_xx(pred_spans_cxw)
        gt_xx = span_cxw_to_xx(gt_spans_cxw)

        desired_start_delta = gt_xx[:, 0] - pred_xx[:, 0]
        desired_end_delta = gt_xx[:, 1] - pred_xx[:, 1]

        # Convert to clip units
        desired_start_clips = desired_start_delta / self.unit
        desired_end_clips = desired_end_delta / self.unit

        # Use step-specific deltas
        deltas = self._get_deltas(step)
        oracle_start = (desired_start_clips.unsqueeze(-1) - deltas.unsqueeze(0)).abs().argmin(dim=-1)
        oracle_end = (desired_end_clips.unsqueeze(-1) - deltas.unsqueeze(0)).abs().argmin(dim=-1)

        return oracle_start, oracle_end

    def supervised_loss(self, state, pred_spans_cxw, gt_spans_cxw, hidden=None, step=0):
        """Compute supervised cross-entropy loss toward oracle actions at a given step.

        Returns:
            loss: scalar
            info: dict with loss components
        """
        start_logits, end_logits, value, new_hidden, iou_pred = self(state, hidden, step=step)
        oracle_start, oracle_end = self.compute_oracle_actions(pred_spans_cxw, gt_spans_cxw, step=step)

        start_loss = F.cross_entropy(start_logits, oracle_start)
        end_loss = F.cross_entropy(end_logits, oracle_end)
        loss = start_loss + end_loss

        info = {
            'start_loss': start_loss.item(),
            'end_loss': end_loss.item(),
        }

        with torch.no_grad():
            start_acc = (start_logits.argmax(dim=-1) == oracle_start).float().mean().item()
            end_acc = (end_logits.argmax(dim=-1) == oracle_end).float().mean().item()
            # Also log oracle action distribution for diagnostics
            no_change_frac = ((oracle_start == self.NUM_ACTIONS // 2) &
                              (oracle_end == self.NUM_ACTIONS // 2)).float().mean().item()
            info['start_acc'] = start_acc
            info['end_acc'] = end_acc
            info['no_change_frac'] = no_change_frac

        if self.use_iou_aux and iou_pred is not None:
            actual_iou = compute_iou_reward(pred_spans_cxw, gt_spans_cxw)
            iou_loss = F.mse_loss(iou_pred, actual_iou)
            loss = loss + 0.5 * iou_loss
            info['iou_aux_loss'] = iou_loss.item()

        return loss, info, new_hidden


# =============================================================================
# Span Refiner (supports both continuous and discrete)
# =============================================================================

class RLSpanRefiner(nn.Module):
    """
    Wraps the pretrained VideoLights model + RL refinement policy.
    Supports both continuous (SpanRefinementPolicy) and discrete (DiscreteSpanPolicy).
    """

    def __init__(self, base_model, policy, num_refine_steps=3, use_gate_attn=False):
        super().__init__()
        self.base_model = base_model
        self.policy = policy
        self.num_refine_steps = num_refine_steps
        self.use_gate_attn = use_gate_attn
        self.is_discrete = isinstance(policy, DiscreteSpanPolicy)

        for param in self.base_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def get_base_predictions(self, model_inputs):
        self.base_model.eval()
        outputs = self.base_model(**model_inputs)
        return outputs

    def build_state(self, pred_spans, pred_scores, vid_mem, txt_mem, vid_mask, txt_mask):
        """Build state with span-local video features, boundary features, and optional gate attention.

        State components:
          - pred_spans (2): current center, width
          - pred_scores (1): confidence score
          - vid_local (D): Gaussian-weighted video features for span context
          - start_feat (D): video features at start boundary
          - end_feat (D): video features at end boundary
          - txt_pooled (D): pooled text features
        """
        bsz, num_queries, _ = pred_spans.shape
        L_vid = vid_mem.shape[1]
        D = vid_mem.shape[2]

        # Span-local video features: Gaussian attention centered at each span
        time_grid = torch.linspace(0, 1, L_vid, device=vid_mem.device)
        span_center = pred_spans[:, :, 0].unsqueeze(-1)
        span_width = pred_spans[:, :, 1].unsqueeze(-1).clamp(min=0.01)
        time_grid_exp = time_grid.view(1, 1, -1).expand(bsz, num_queries, -1)
        weights = torch.exp(-0.5 * ((time_grid_exp - span_center) / (span_width * 0.5 + 1e-6)) ** 2)
        vid_mask_exp = vid_mask.float().unsqueeze(1).expand(-1, num_queries, -1)
        weights = weights * vid_mask_exp
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        vid_local = torch.einsum('bql,bld->bqd', weights, vid_mem)

        # Boundary features: video features at start/end positions
        # This gives the agent direct information about what's at each boundary
        start_pos = (span_center.squeeze(-1) - span_width.squeeze(-1) / 2).clamp(0.0, 1.0)
        end_pos = (span_center.squeeze(-1) + span_width.squeeze(-1) / 2).clamp(0.0, 1.0)

        # Soft boundary features: use narrow Gaussian at boundary positions (sigma=2 clips)
        boundary_sigma = 2.0 / L_vid
        start_weights = torch.exp(-0.5 * ((time_grid_exp - start_pos.unsqueeze(-1)) / (boundary_sigma + 1e-6)) ** 2)
        end_weights = torch.exp(-0.5 * ((time_grid_exp - end_pos.unsqueeze(-1)) / (boundary_sigma + 1e-6)) ** 2)
        start_weights = start_weights * vid_mask_exp
        end_weights = end_weights * vid_mask_exp
        start_weights = start_weights / (start_weights.sum(dim=-1, keepdim=True) + 1e-8)
        end_weights = end_weights / (end_weights.sum(dim=-1, keepdim=True) + 1e-8)
        start_feat = torch.einsum('bql,bld->bqd', start_weights, vid_mem)  # (bsz, nq, D)
        end_feat = torch.einsum('bql,bld->bqd', end_weights, vid_mem)      # (bsz, nq, D)

        txt_mask_f = txt_mask.float().unsqueeze(-1)
        txt_pooled = (txt_mem * txt_mask_f).sum(dim=1) / txt_mask_f.sum(dim=1).clamp(min=1)
        txt_pooled = txt_pooled.unsqueeze(1).expand(-1, num_queries, -1)

        # Gate attention: text gates video features (TSP-PRL style)
        if self.use_gate_attn and hasattr(self.policy, 'gate_proj'):
            gate = torch.sigmoid(self.policy.gate_proj(txt_pooled))
            vid_local = vid_local * gate

        state = torch.cat([
            pred_spans,
            pred_scores.unsqueeze(-1),
            vid_local,
            start_feat,
            end_feat,
            txt_pooled,
        ], dim=-1)

        state = state.view(bsz * num_queries, -1)
        return state

    def refine_spans(self, pred_spans, state, hidden=None, deterministic=False, step=0):
        """Apply one step of RL refinement. Works for both continuous and discrete."""
        if self.is_discrete:
            action, log_prob, entropy, value, new_hidden, iou_pred = \
                self.policy.get_action_and_value(state, hidden=hidden, deterministic=deterministic, step=step)
            delta = self.policy.action_to_delta(action, step=step)
        else:
            action, log_prob, entropy, value = \
                self.policy.get_action_and_value(state, deterministic=deterministic)
            delta = action
            new_hidden = None
            iou_pred = None

        refined_spans = pred_spans + delta
        refined_center = refined_spans[:, 0].clamp(0.0, 1.0)
        refined_width = refined_spans[:, 1].clamp(0.01, 1.0)
        refined_spans = torch.stack([refined_center, refined_width], dim=-1)

        return refined_spans, action, log_prob, entropy, value, new_hidden, iou_pred

    def forward_rl(self, model_inputs, targets=None, deterministic=False):
        """Full RL forward pass: base model -> multi-step refinement."""
        with torch.no_grad():
            base_outputs = self.get_base_predictions(model_inputs)

        pred_spans = base_outputs['pred_spans'].detach()
        pred_logits = base_outputs['pred_logits'].detach()
        pred_scores = F.softmax(pred_logits, dim=-1)[..., 0]

        vid_mem = base_outputs['src_vid'].detach()
        txt_mem = base_outputs['src_txt'].detach()
        vid_mask = model_inputs['src_vid_mask']
        txt_mask = model_inputs['src_txt_mask']

        bsz, num_queries, _ = pred_spans.shape

        all_actions = []
        all_log_probs = []
        all_entropies = []
        all_values = []
        all_iou_preds = []
        all_spans = [pred_spans.view(bsz * num_queries, 2)]

        current_spans = pred_spans.view(bsz * num_queries, 2)
        hidden = None

        for step in range(self.num_refine_steps):
            current_spans_reshaped = current_spans.view(bsz, num_queries, 2)
            state = self.build_state(
                current_spans_reshaped, pred_scores,
                vid_mem, txt_mem, vid_mask, txt_mask
            )

            refined_spans, action, log_prob, entropy, value, hidden, iou_pred = \
                self.refine_spans(current_spans, state, hidden=hidden, deterministic=deterministic, step=step)

            all_actions.append(action)
            all_log_probs.append(log_prob)
            all_entropies.append(entropy)
            all_values.append(value)
            all_iou_preds.append(iou_pred)
            all_spans.append(refined_spans)

            current_spans = refined_spans

        final_spans = current_spans.view(bsz, num_queries, 2)

        return {
            'initial_spans': pred_spans,
            'refined_spans': final_spans,
            'pred_logits': pred_logits,
            'pred_scores': pred_scores,
            'all_actions': all_actions,
            'all_log_probs': all_log_probs,
            'all_entropies': all_entropies,
            'all_values': all_values,
            'all_iou_preds': all_iou_preds,
            'all_spans': all_spans,
            'saliency_scores': base_outputs.get('saliency_scores', None),
            'base_outputs': base_outputs,
        }


# =============================================================================
# Reward computation
# =============================================================================

def compute_iou_reward(pred_spans_cxw, target_spans_cxw, clip_len=2, max_v_l=75):
    """Compute IoU-based reward for span predictions."""
    pred_xx = span_cxw_to_xx(pred_spans_cxw)
    target_xx = span_cxw_to_xx(target_spans_cxw)

    pred_xx = pred_xx.clamp(min=0.0, max=1.0)
    target_xx = target_xx.clamp(min=0.0, max=1.0)

    inter_start = torch.max(pred_xx[:, 0], target_xx[:, 0])
    inter_end = torch.min(pred_xx[:, 1], target_xx[:, 1])
    inter = (inter_end - inter_start).clamp(min=0)

    pred_len = (pred_xx[:, 1] - pred_xx[:, 0]).clamp(min=1e-6)
    target_len = (target_xx[:, 1] - target_xx[:, 0]).clamp(min=1e-6)
    union = pred_len + target_len - inter

    iou = inter / union.clamp(min=1e-6)
    return iou


# =============================================================================
# PPO Trainer (supports both continuous and discrete)
# =============================================================================

class PPOTrainer:
    """PPO trainer for the RL span refinement agent."""

    def __init__(self, policy, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, entropy_coef=0.01, value_coef=0.5,
                 max_grad_norm=0.5, ppo_epochs=4, iou_aux_coef=0.0,
                 reward_scale=10.0, direction_coef=5.0, terminal_bonus=5.0):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.iou_aux_coef = iou_aux_coef
        self.reward_scale = reward_scale
        self.direction_coef = direction_coef
        self.terminal_bonus = terminal_bonus
        self.is_discrete = isinstance(policy, DiscreteSpanPolicy)

    def compute_rewards(self, all_spans, target_spans_flat, num_steps):
        """Compute per-step shaped rewards with direction bonus and terminal bonus.

        Combines:
          1. Delta IoU: IoU improvement from previous step (scaled)
          2. Direction reward: boundary distance reduction toward GT
          3. Terminal bonus: extra reward at last step for overall improvement
        """
        rewards = []
        initial_iou = compute_iou_reward(all_spans[0], target_spans_flat)
        prev_iou = initial_iou

        # Pre-compute GT boundaries for direction reward
        gt_xx = span_cxw_to_xx(target_spans_flat).clamp(0.0, 1.0)

        for step in range(1, num_steps + 1):
            curr_iou = compute_iou_reward(all_spans[step], target_spans_flat)
            delta_iou = curr_iou - prev_iou

            # Direction reward: reduction in L1 distance to GT boundaries
            prev_xx = span_cxw_to_xx(all_spans[step - 1]).clamp(0.0, 1.0)
            curr_xx = span_cxw_to_xx(all_spans[step]).clamp(0.0, 1.0)
            dist_before = (prev_xx - gt_xx).abs().sum(dim=-1)
            dist_after = (curr_xx - gt_xx).abs().sum(dim=-1)
            direction_reward = dist_before - dist_after  # positive if moved closer

            reward = self.reward_scale * delta_iou + self.direction_coef * direction_reward

            # Terminal bonus for overall improvement at last step
            if step == num_steps:
                total_improvement = curr_iou - initial_iou
                reward = reward + self.terminal_bonus * torch.clamp(total_improvement, min=0.0)

            rewards.append(reward)
            prev_iou = curr_iou

        return rewards

    def compute_gae(self, rewards, values, dones=None):
        """Compute Generalized Advantage Estimation."""
        T = len(rewards)
        N = rewards[0].shape[0]
        device = rewards[0].device

        advantages = torch.zeros(T, N, device=device)
        last_gae = torch.zeros(N, device=device)

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = torch.zeros(N, device=device)
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * last_gae
            advantages[t] = last_gae

        returns = advantages + torch.stack(values)
        return advantages, returns

    def update(self, states_list, actions_list, old_log_probs_list,
               rewards, values_list, entropies_list,
               iou_targets_list=None, iou_preds_list=None):
        """PPO update step. Supports both continuous and discrete policies.
        For discrete+GRU, processes timesteps sequentially to maintain hidden state."""
        T = len(rewards)

        values_detached = [v.detach() for v in values_list]
        advantages, returns = self.compute_gae(rewards, values_detached)

        # IoU auxiliary targets
        has_iou_aux = (self.iou_aux_coef > 0 and iou_targets_list is not None
                       and iou_preds_list is not None
                       and iou_preds_list[0] is not None)
        if has_iou_aux:
            iou_targets_per_step = [t.detach() for t in iou_targets_list]

        use_sequential = (self.is_discrete and hasattr(self.policy, 'use_gru')
                          and self.policy.use_gru)

        total_policy_loss = 0.
        total_value_loss = 0.
        total_entropy_loss = 0.
        total_iou_loss = 0.

        for _ in range(self.ppo_epochs):
            if use_sequential:
                # Process step-by-step to maintain GRU hidden state
                all_new_log_probs = []
                all_entropy = []
                all_new_values = []
                all_new_iou_pred = []
                hidden = None
                for t in range(T):
                    _, new_lp_t, ent_t, val_t, hidden, iou_t = \
                        self.policy.get_action_and_value(
                            states_list[t], action=actions_list[t], hidden=hidden, step=t)
                    all_new_log_probs.append(new_lp_t)
                    all_entropy.append(ent_t)
                    all_new_values.append(val_t)
                    if iou_t is not None:
                        all_new_iou_pred.append(iou_t)
                new_log_probs = torch.cat(all_new_log_probs, dim=0)
                entropy = torch.cat(all_entropy, dim=0)
                new_values = torch.cat(all_new_values, dim=0)
                new_iou_pred = torch.cat(all_new_iou_pred, dim=0) if all_new_iou_pred else None
            elif self.is_discrete:
                # Process per-step since each step has its own action head
                all_new_log_probs = []
                all_entropy = []
                all_new_values = []
                all_new_iou_pred = []
                for t in range(T):
                    _, new_lp_t, ent_t, val_t, _, iou_t = \
                        self.policy.get_action_and_value(
                            states_list[t], action=actions_list[t], step=t)
                    all_new_log_probs.append(new_lp_t)
                    all_entropy.append(ent_t)
                    all_new_values.append(val_t)
                    if iou_t is not None:
                        all_new_iou_pred.append(iou_t)
                new_log_probs = torch.cat(all_new_log_probs, dim=0)
                entropy = torch.cat(all_entropy, dim=0)
                new_values = torch.cat(all_new_values, dim=0)
                new_iou_pred = torch.cat(all_new_iou_pred, dim=0) if all_new_iou_pred else None
            else:
                states = torch.cat(states_list, dim=0)
                actions = torch.cat(actions_list, dim=0)
                _, new_log_probs, entropy, new_values = \
                    self.policy.get_action_and_value(states, action=actions)
                new_iou_pred = None

            old_log_probs_flat = torch.cat(old_log_probs_list, dim=0)
            adv_flat = advantages.reshape(-1)
            ret_flat = returns.reshape(-1)
            adv_norm = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

            ratio = torch.exp(new_log_probs - old_log_probs_flat.detach())
            surr1 = ratio * adv_norm
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_norm
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(new_values, ret_flat.detach())
            entropy_loss = -entropy.mean()

            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

            # IoU auxiliary loss
            if has_iou_aux and new_iou_pred is not None:
                iou_targets_flat = torch.cat(iou_targets_per_step, dim=0)
                iou_loss = F.mse_loss(new_iou_pred, iou_targets_flat)
                loss = loss + self.iou_aux_coef * iou_loss
                total_iou_loss += iou_loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()

        n = self.ppo_epochs
        result = {
            'policy_loss': total_policy_loss / n,
            'value_loss': total_value_loss / n,
            'entropy_loss': total_entropy_loss / n,
        }
        if has_iou_aux:
            result['iou_aux_loss'] = total_iou_loss / n
        return result


# =============================================================================
# REINFORCE Trainer (simpler, trajectory-level reward)
# =============================================================================

class REINFORCETrainer:
    """REINFORCE with baseline for span refinement.

    Uses trajectory-level final IoU as reward (not per-step delta).
    Simpler and more stable than PPO for sparse reward settings.

    Inspired by Read, Watch, and Move (He et al., AAAI 2019) and
    TSP-PRL (Wu et al., AAAI 2020).
    """

    def __init__(self, policy, lr=3e-4, entropy_coef=0.05,
                 max_grad_norm=0.5, baseline_momentum=0.99,
                 iou_aux_coef=0.0):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.baseline_momentum = baseline_momentum
        self.iou_aux_coef = iou_aux_coef
        self.is_discrete = isinstance(policy, DiscreteSpanPolicy)
        self.running_baseline = None

    def compute_trajectory_reward(self, all_spans, target_spans_flat, num_steps):
        """Compute final IoU as the trajectory reward.

        Returns:
            reward: (M,) final IoU value
            improvement: (M,) IoU(final) - IoU(initial)
        """
        initial_iou = compute_iou_reward(all_spans[0], target_spans_flat)
        final_iou = compute_iou_reward(all_spans[-1], target_spans_flat)
        improvement = final_iou - initial_iou
        return final_iou, improvement

    def update(self, log_probs_per_step, entropies_per_step, reward,
               iou_targets_list=None, iou_preds_list=None,
               states_list=None, actions_list=None):
        """REINFORCE update with trajectory reward and exponential moving average baseline.

        Args:
            log_probs_per_step: list of (M,) log probs at each step
            entropies_per_step: list of (M,) entropies at each step
            reward: (M,) final IoU reward for the trajectory
            iou_targets_list: optional list of (M,) IoU targets per step
            iou_preds_list: optional list of (M,) IoU predictions per step
        """
        # Update baseline
        reward_mean = reward.mean().item()
        if self.running_baseline is None:
            self.running_baseline = reward_mean
        else:
            self.running_baseline = (self.baseline_momentum * self.running_baseline
                                     + (1 - self.baseline_momentum) * reward_mean)

        advantage = reward - self.running_baseline

        # Sum log probs across steps (trajectory-level)
        total_log_prob = torch.stack(log_probs_per_step, dim=0).sum(dim=0)  # (M,)
        total_entropy = torch.stack(entropies_per_step, dim=0).mean(dim=0)  # (M,)

        policy_loss = -(total_log_prob * advantage.detach()).mean()
        entropy_loss = -total_entropy.mean()

        loss = policy_loss + self.entropy_coef * entropy_loss

        # IoU auxiliary loss
        has_iou_aux = (self.iou_aux_coef > 0 and iou_targets_list is not None
                       and iou_preds_list is not None
                       and iou_preds_list[0] is not None)
        iou_loss_val = 0.0
        if has_iou_aux:
            iou_targets_flat = torch.cat([t.detach() for t in iou_targets_list], dim=0)
            iou_preds_flat = torch.cat([p for p in iou_preds_list if p is not None], dim=0)
            iou_loss = F.mse_loss(iou_preds_flat, iou_targets_flat)
            loss = loss + self.iou_aux_coef * iou_loss
            iou_loss_val = iou_loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        result = {
            'policy_loss': policy_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'baseline': self.running_baseline,
            'advantage_mean': advantage.mean().item(),
        }
        if has_iou_aux:
            result['iou_aux_loss'] = iou_loss_val
        return result
