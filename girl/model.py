# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from girl.feature_refinement import FeatureRefinement
from girl.components import GlobalFeatureExtractor, ConvLinearLayer, MLP, LinearLayer
from girl.span_utils import generalized_temporal_iou, span_cxw_to_xx

from girl.matcher import build_matcher
from girl.transformer import build_transformer
from girl.position_encoding import build_position_encoding
from girl.misc import accuracy
import numpy as np


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class VideoLight(nn.Module):
    """ QD DETR. """

    def __init__(self, transformer, position_embed, txt_position_embed, txt_dim, vid_dim,
                 num_queries, input_dropout, aux_loss=False, mr_to_hd_loss=False, fra=True,
                 contrastive_align_loss=False, contrastive_hdim=64,
                 max_v_l=75, span_loss_type="l1", use_txt_pos=False, n_input_proj=2, aud_dim=0, clip_len=2):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            position_embed: torch module of the position_embedding, See position_encoding.py
            txt_position_embed: position_embedding for text
            txt_dim: int, text query input dimension
            vid_dim: int, video feature input dimension
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         QD-DETR can detect in a single video.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            contrastive_align_loss: If true, perform span - tokens contrastive learning
            contrastive_hdim: dimension used for projecting the embeddings before computing contrastive loss
            max_v_l: int, maximum #clips in videos
            span_loss_type: str, one of [l1, ce]
                l1: (center-x, width) regression.
                ce: (st_idx, ed_idx) classification.
            # foreground_thd: float, intersection over prediction >= foreground_thd: labeled as foreground
            # background_thd: float, intersection over prediction <= background_thd: labeled background
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        hidden_dim = transformer.d_model
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        span_pred_dim = 2 if span_loss_type == "l1" else max_v_l * 2
        self.span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
        self.class_embed = nn.Linear(hidden_dim, 2)  # 0: background, 1: foreground
        # IoU-aware prediction head (Rank-DETR inspired)
        self.iou_embed = MLP(hidden_dim, hidden_dim, 1, 3)
        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj
        self.query_embed = nn.Embedding(num_queries, 2)
        relu_args = [True] * 3
        relu_args[n_input_proj - 1] = False
        self.fra = fra
        if self.fra :
            self.input_txt_proj = nn.Sequential(*[
                                                     ConvLinearLayer(txt_dim, hidden_dim, layer_norm=True,
                                                                     dropout=input_dropout, relu=relu_args[0]),
                                                     ConvLinearLayer(hidden_dim, hidden_dim, layer_norm=True,
                                                                     dropout=input_dropout, relu=relu_args[1]),
                                                     ConvLinearLayer(hidden_dim, hidden_dim, layer_norm=True,
                                                                     dropout=input_dropout, relu=relu_args[2])
                                                 ][:n_input_proj])
            self.input_vid_proj = nn.Sequential(*[
                                                     ConvLinearLayer(vid_dim + aud_dim, hidden_dim, layer_norm=True,
                                                                     dropout=input_dropout, relu=relu_args[0]),
                                                     ConvLinearLayer(hidden_dim, hidden_dim, layer_norm=True,
                                                                     dropout=input_dropout, relu=relu_args[1]),
                                                     ConvLinearLayer(hidden_dim, hidden_dim, layer_norm=True,
                                                                     dropout=input_dropout, relu=relu_args[2])
                                                 ][:n_input_proj])
        else:
            self.input_txt_proj = nn.Sequential(*[
                                                     LinearLayer(txt_dim, hidden_dim, layer_norm=True,
                                                                     dropout=input_dropout, relu=relu_args[0]),
                                                     LinearLayer(hidden_dim, hidden_dim, layer_norm=True,
                                                                     dropout=input_dropout, relu=relu_args[1]),
                                                     LinearLayer(hidden_dim, hidden_dim, layer_norm=True,
                                                                     dropout=input_dropout, relu=relu_args[2])
                                                 ][:n_input_proj])
            self.input_vid_proj = nn.Sequential(*[
                                                     LinearLayer(vid_dim + aud_dim, hidden_dim, layer_norm=True,
                                                                     dropout=input_dropout, relu=relu_args[0]),
                                                     LinearLayer(hidden_dim, hidden_dim, layer_norm=True,
                                                                     dropout=input_dropout, relu=relu_args[1]),
                                                     LinearLayer(hidden_dim, hidden_dim, layer_norm=True,
                                                                     dropout=input_dropout, relu=relu_args[2])
                                                 ][:n_input_proj])

        self.contrastive_align_loss = contrastive_align_loss
        if contrastive_align_loss:
            self.contrastive_align_projection_query = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_txt = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_vid = nn.Linear(hidden_dim, contrastive_hdim)


        if self.fra:
            self.feature_refinement = FeatureRefinement(hidden_dim)
        self.saliency_proj1 = nn.Linear(hidden_dim, hidden_dim)
        self.saliency_proj2 = nn.Linear(hidden_dim, hidden_dim)
        self.aux_loss = aux_loss

        self.hidden_dim = hidden_dim
        self.global_rep_token = torch.nn.Parameter(torch.randn(hidden_dim))
        self.global_rep_pos = torch.nn.Parameter(torch.randn(hidden_dim))

        self.mr_to_hd_loss = mr_to_hd_loss
        if mr_to_hd_loss:
            # self.weighted_pool = WeightedPool(hidden_dim)
            self.gru_extractor = GlobalFeatureExtractor(hidden_dim, hidden_dim, num_layers=1, bidirectional=False)
            self.saliency_proj_mr = nn.Linear(hidden_dim, hidden_dim)
            # self.saliency_proj1_mr = nn.Linear(hidden_dim, hidden_dim)
            # self.saliency_proj2_mr = nn.Linear(hidden_dim, hidden_dim)
        self.clip_len = clip_len

    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask, src_aud=None, src_aud_mask=None):
        """The forward expects two tensors:
               - src_txt: [batch_size, L_txt, D_txt]
               - src_txt_mask: [batch_size, L_txt], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer
               - src_vid: [batch_size, L_vid, D_vid]
               - src_vid_mask: [batch_size, L_vid], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer

            It returns a dict with the following elements:
               - "pred_spans": The normalized boxes coordinates for all queries, represented as
                               (center_x, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if src_aud is not None:
            src_vid = torch.cat([src_vid, src_aud], dim=2)

        src_vid = self.input_vid_proj(src_vid)
        src_txt = self.input_txt_proj(src_txt)

        src_vid_ed = src_vid

        if self.fra:
            src_vid = self.feature_refinement(src_vid, src_txt, src_vid_mask, src_txt_mask)

        src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()  # (bsz, L_vid+L_txt)
        # TODO should we remove or use different positional embeddings to the src_txt?
        pos_vid = self.position_embed(src_vid, src_vid_mask)  # (bsz, L_vid, d)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d)
        # pos_txt = torch.zeros_like(src_txt)
        # pad zeros for txt positions
        pos = torch.cat([pos_vid, pos_txt], dim=1)
        # (#layers, bsz, #queries, d), (bsz, L_vid+L_txt, d)

        # for global token
        mask_ = torch.tensor([[True]]).to(mask.device).repeat(mask.shape[0], 1)
        mask = torch.cat([mask_, mask], dim=1)
        src_ = self.global_rep_token.reshape([1, 1, self.hidden_dim]).repeat(src.shape[0], 1, 1)
        src = torch.cat([src_, src], dim=1)
        pos_ = self.global_rep_pos.reshape([1, 1, self.hidden_dim]).repeat(pos.shape[0], 1, 1)
        pos = torch.cat([pos_, pos], dim=1)

        video_length = src_vid.shape[1]

        hs, reference, memory, memory_global = self.transformer(src, ~mask, self.query_embed.weight, pos,
                                                                video_length=video_length)
        outputs_class = self.class_embed(hs)  # (#layers, batch_size, #queries, #classes)
        reference_before_sigmoid = inverse_sigmoid(reference)
        tmp = self.span_embed(hs)
        outputs_coord = tmp + reference_before_sigmoid
        if self.span_loss_type == "l1":
            outputs_coord = outputs_coord.sigmoid()
        # IoU-aware prediction: predict IoU quality for each query
        outputs_iou = self.iou_embed(hs).sigmoid().squeeze(-1)  # (#layers, bsz, #queries)
        out = {'pred_logits': outputs_class[-1], 'pred_spans': outputs_coord[-1],
               'pred_iou': outputs_iou[-1]}

        txt_mem = memory[:, src_vid.shape[1]:]  # (bsz, L_txt, d)
        vid_mem = memory[:, :src_vid.shape[1]]  # (bsz, L_vid, d)
        if self.contrastive_align_loss:
            proj_queries = F.normalize(self.contrastive_align_projection_query(hs), p=2, dim=-1)
            proj_txt_mem = F.normalize(self.contrastive_align_projection_txt(src_txt), p=2, dim=-1)
            proj_vid_mem = F.normalize(self.contrastive_align_projection_vid(vid_mem), p=2, dim=-1)
            out.update(dict(
                proj_queries=proj_queries[-1],
                proj_txt_mem=proj_txt_mem,
                proj_vid_mem=proj_vid_mem
            ))

        # !!! this is code for test
        if src_txt.shape[1] == 0:
            print("There is zero text query. You should change codes properly")
            exit(-1)

        ### Neg Pairs ###
        src_txt_neg = torch.cat([src_txt[1:], src_txt[0:1]], dim=0)
        src_txt_mask_neg = torch.cat([src_txt_mask[1:], src_txt_mask[0:1]], dim=0)
        src_neg = torch.cat([src_vid, src_txt_neg], dim=1)
        mask_neg = torch.cat([src_vid_mask, src_txt_mask_neg], dim=1).bool()

        mask_neg = torch.cat([mask_, mask_neg], dim=1)
        src_neg = torch.cat([src_, src_neg], dim=1)
        pos_neg = pos.clone()  # since it does not use actual content

        _, _, memory_neg, memory_global_neg = self.transformer(src_neg, ~mask_neg, self.query_embed.weight, pos_neg,
                                                               video_length=video_length)
        vid_mem_neg = memory_neg[:, :src_vid.shape[1]]

        out["saliency_scores"] = (
                torch.sum(self.saliency_proj1(vid_mem) * self.saliency_proj2(memory_global).unsqueeze(1),
                          dim=-1) / np.sqrt(self.hidden_dim))

        out["saliency_scores_neg"] = (
                torch.sum(self.saliency_proj1(vid_mem_neg) * self.saliency_proj2(memory_global_neg).unsqueeze(1),
                          dim=-1) / np.sqrt(self.hidden_dim))

        if self.mr_to_hd_loss:
            out["saliency_scores_mr"] = self.get_saliency_from_mr(src_vid_ed, src_vid_mask, vid_mem,
                                                                  outputs_class[-1], outputs_coord[-1])

        # print(src_vid_mask.shape, src_vid.shape, vid_mem_neg.shape, vid_mem.shape)
        out["video_mask"] = src_vid_mask
        out["src_txt"] = src_txt
        out["src_vid"] = src_vid

        if self.aux_loss:
            out['aux_outputs'] = [
                {'pred_logits': a, 'pred_spans': b, 'pred_iou': c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_iou[:-1])]
            if self.contrastive_align_loss:
                assert proj_queries is not None
                for idx, d in enumerate(proj_queries[:-1]):
                    out['aux_outputs'][idx].update(dict(proj_queries=d, proj_txt_mem=proj_txt_mem,
                                                        proj_vid_mem=proj_vid_mem))
        return out

    def get_saliency_from_mr(self, src_vid, vid_mask, memory, pred_logits, pred_spans):

        video_length = src_vid.shape[1]

        prob = F.softmax(pred_logits, -1)
        scores = prob[..., 0]
        sorted_scores, sorted_indices = torch.sort(scores, dim=-1, descending=True)
        sorted_indices_max = sorted_indices[:, :1]

        spans = span_cxw_to_xx(pred_spans) * (video_length * self.clip_len)
        spans = torch.floor(spans / self.clip_len)

        selected_values_max = spans[torch.arange(spans.size(0)).unsqueeze(1), sorted_indices_max].squeeze(1)

        sliced_samples = []
        b = memory.size(0)
        max_time = memory.size(1)

        fixed_slice_size = max_time

        for i in range(b):
            start_time = int(selected_values_max[i, 0])
            end_time = int(selected_values_max[i, 1])
            sliced_sample = src_vid[i, start_time:end_time + 1, :]

            padding_size = fixed_slice_size - sliced_sample.size(0)
            if padding_size > 0:
                padded_slice = F.pad(sliced_sample, (0, 0, 0, padding_size), value=0)
            else:
                padded_slice = sliced_sample[:fixed_slice_size, :]

            sliced_samples.append(padded_slice)

        sliced_features = torch.stack(sliced_samples, dim=0)  # torch.Size([32, 75, 256])

        # global_features = self.weighted_pool(sliced_features, vid_mask).unsqueeze(1)  # torch.Size([32, 256])
        global_features = self.gru_extractor(sliced_features).unsqueeze(1)  # torch.Size([32, 256])
        weight = torch.matmul(global_features, src_vid.transpose(1, 2)).squeeze(1)  # shape: (32, 1, 75)
        memory = memory * weight.unsqueeze(-1) + memory
        saliency_scores = (torch.sum(self.saliency_proj_mr(memory), dim=-1) / np.sqrt(self.hidden_dim))
        # saliency_scores = (
        #             torch.sum(self.saliency_proj1(memory) * self.saliency_proj2(global_features).unsqueeze(1),
        #                       dim=-1) / np.sqrt(self.hidden_dim))
        return saliency_scores

    # @torch.jit.unused
    # def _set_aux_loss(self, outputs_class, outputs_coord):
    #     # this is a workaround to make torchscript happy, as torchscript
    #     # doesn't support dictionary with non-homogeneous values, such
    #     # as a dict having both a Tensor and a list.
    #     return [{'pred_logits': a, 'pred_spans': b}
    #             for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, weight_dict, eos_coef, losses, temperature, span_loss_type, max_v_l,
                 saliency_margin=1, use_matcher=True, n_epoch=200, hard_pos_neg_loss=False,
                 hard_pos_neg_loss_coef=10.0, clip_len=2, mr_to_hd_loss=False, mr_to_hd_loss_coef=1.0, cos_sim_loss_coef=1.0):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            temperature: float, temperature for NCE loss
            span_loss_type: str, [l1, ce]
            max_v_l: int,
            saliency_margin: float
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.temperature = temperature
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.saliency_margin = saliency_margin

        # foreground and background classification
        self.foreground_label = 0
        self.background_label = 1
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
        self.register_buffer('empty_weight', empty_weight)

        # for tvsum,
        self.use_matcher = use_matcher
        self.n_epoch = n_epoch
        self.hard_pos_neg_loss = hard_pos_neg_loss
        self.hard_pos_neg_loss_coef = hard_pos_neg_loss_coef
        self.cos_sim_loss_coef = cos_sim_loss_coef
        self.clip_len = clip_len
        self.mr_to_hd_loss = mr_to_hd_loss
        self.mr_to_hd_loss_coef = mr_to_hd_loss_coef

    def loss_spans(self, outputs, targets, indices, epoch_i=0):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
           The target spans are expected in format (center_x, w), normalized by the image size.
        """
        assert 'pred_spans' in outputs
        targets = targets["span_labels"]
        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs['pred_spans'][idx]  # (#spans, max_v_l * 2)
        tgt_spans = torch.cat([t['spans'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # (#spans, 2)
        if self.span_loss_type == "l1":
            loss_span = F.l1_loss(src_spans, tgt_spans, reduction='none')
            loss_giou = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))
        else:  # ce
            n_spans = src_spans.shape[0]
            src_spans = src_spans.view(n_spans, 2, self.max_v_l).transpose(1, 2)
            loss_span = F.cross_entropy(src_spans, tgt_spans, reduction='none')

            # giou
            # src_span_indices = src_spans.max(1)[1]  # (#spans, 2)
            # src_span_indices[:, 1] += 1  # ed non-inclusive [st, ed)
            #
            # tgt_span_indices = tgt_spans
            # tgt_span_indices[:, 1] += 1
            # loss_giou = 1 - torch.diag(generalized_temporal_iou(src_span_indices, tgt_span_indices))
            loss_giou = loss_span.new_zeros([1])

        losses = {}
        losses['loss_span'] = loss_span.mean()
        losses['loss_giou'] = loss_giou.mean()
        return losses

    def loss_iou(self, outputs, targets, indices, epoch_i=0):
        """IoU-aware quality prediction loss.
        Supervise the predicted IoU score with the actual IoU between prediction and GT.
        This aligns the confidence score with localization quality (Rank-DETR inspired).
        """
        if 'pred_iou' not in outputs:
            return {'loss_iou': torch.tensor(0.0, device=outputs['pred_spans'].device)}

        targets = targets["span_labels"]
        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs['pred_spans'][idx].detach()  # (#matched, 2) -- detach: IoU head doesn't affect span
        tgt_spans = torch.cat([t['spans'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # Compute actual IoU as supervision target
        if self.span_loss_type == "l1":
            with torch.no_grad():
                actual_iou = torch.diag(generalized_temporal_iou(
                    span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)
                )).clamp(min=0.0)  # (#matched,)

        pred_iou = outputs['pred_iou'][idx]  # (#matched,)
        loss = F.l1_loss(pred_iou, actual_iou, reduction='mean')
        return {'loss_iou': loss}

    def loss_labels(self, outputs, targets, indices, log=True, epoch_i=0):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # TODO add foreground and background classifier.  use all non-matched as background.
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # (batch_size, #queries, #classes=2)
        # idx is a tuple of two 1D tensors (batch_idx, src_idx), of the same length == #objects in batch
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.background_label,
                                    dtype=torch.int64, device=src_logits.device)  # (batch_size, #queries)
        target_classes[idx] = self.foreground_label

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction="none")
        losses = {'loss_label': loss_ce.mean()}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], self.foreground_label)[0]
        return losses

    def adaptive_hard_neg_loss_saliency(self, saliency_scores, gt_tensor, targets, indices, epoch_i=0):
        """higher scores for positive clips"""
        if "saliency_pos_index" not in targets:
            return 0
        saliency_pos_index = targets["saliency_pos_index"]
        res_tensor = saliency_scores * ~saliency_pos_index.bool()
        gt_tensor = gt_tensor * ~saliency_pos_index.bool()
        hard_negatives_loss = F.mse_loss(gt_tensor, res_tensor)
        hard_negatives_loss = hard_negatives_loss * (epoch_i + 1) * self.hard_pos_neg_loss_coef
        return hard_negatives_loss

    def adaptive_hard_pos_loss_saliency(self, saliency_scores, gt_tensor, targets, indices, epoch_i=0):
        """higher scores for positive clips"""
        if "saliency_pos_index" not in targets:
            return 0
        saliency_pos_index = targets["saliency_pos_index"]
        res_tensor = saliency_scores * saliency_pos_index.bool()
        gt_tensor = gt_tensor * saliency_pos_index.bool()
        hard_positive_loss = F.mse_loss(gt_tensor, res_tensor)
        hard_positive_loss = hard_positive_loss * (epoch_i + 1) * self.hard_pos_neg_loss_coef
        return hard_positive_loss

    def loss_saliency(self, outputs, targets, indices, log=True, epoch_i=0):
        """higher scores for positive clips"""
        if "saliency_pos_labels" not in targets:
            return {"loss_saliency": 0}

        vid_token_mask = outputs["video_mask"]

        # Neg pair loss
        saliency_scores_neg = outputs["saliency_scores_neg"].clone()  # (N, L)
        # loss_neg_pair = torch.sigmoid(saliency_scores_neg).mean()

        loss_neg_pair = (- torch.log(1. - torch.sigmoid(saliency_scores_neg)) * vid_token_mask).sum(dim=1).mean()

        saliency_scores = outputs["saliency_scores"].clone()  # (N, L)
        saliency_contrast_label = targets["saliency_all_labels"]

        saliency_scores = torch.cat([saliency_scores, saliency_scores_neg], dim=1)
        saliency_contrast_label = torch.cat([saliency_contrast_label, torch.zeros_like(saliency_contrast_label)], dim=1)

        vid_token_mask = vid_token_mask.repeat([1, 2])
        saliency_scores = vid_token_mask * saliency_scores + (1. - vid_token_mask) * -1e+3

        tau = 0.5
        loss_rank_contrastive = 0.

        # for rand_idx in range(1, 13, 3):
        #     # 1, 4, 7, 10 --> 5 stages
        for rand_idx in range(1, 12):
            drop_mask = ~(saliency_contrast_label > 100)  # no drop
            pos_mask = (saliency_contrast_label >= rand_idx)  # positive when equal or higher than rand_idx

            if torch.sum(pos_mask) == 0:  # no positive sample
                continue
            else:
                batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

            # drop higher ranks
            cur_saliency_scores = saliency_scores * drop_mask / tau + ~drop_mask * -1e+3

            # numerical stability
            logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]

            # softmax
            exp_logits = torch.exp(logits)
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

            mean_log_prob_pos = (pos_mask * log_prob * vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)

            loss = - mean_log_prob_pos * batch_drop_mask

            loss_rank_contrastive = loss_rank_contrastive + loss.mean()

        loss_rank_contrastive = loss_rank_contrastive / 12

        saliency_scores = outputs["saliency_scores"]  # (N, L)
        pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
        neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
        num_pairs = pos_indices.shape[1]  # typically 2 or 4
        batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
        pos_scores = torch.stack(
            [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        neg_scores = torch.stack(
            [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        loss_saliency = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
                        / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale

        gt_tensor = targets["saliency_all_labels"]
        saliency_scores = outputs["saliency_scores"]  # (N, L)
        saliency_scores = ((saliency_scores - saliency_scores.min()) /
                           (gt_tensor.max() - gt_tensor.min()))

        gt_norm = F.normalize(gt_tensor, dim=1, p=2)
        sal_norm = F.normalize(saliency_scores, dim=1, p=2)
        cos_sim = F.cosine_similarity(gt_norm, sal_norm)
        cos_sim_err = (1 - cos_sim) * self.cos_sim_loss_coef

        sal_from_mr_cos_sim_loss = torch.zeros(1).to(saliency_scores.device)
        if self.mr_to_hd_loss and "saliency_scores_mr" in outputs:
            sal_from_mr = outputs["saliency_scores_mr"]
            sal_from_mr = ((sal_from_mr - sal_from_mr.min()) /
                           (gt_tensor.max() - gt_tensor.min()))
            sal_from_mr = F.normalize(sal_from_mr, dim=1, p=2)
            sal_from_mr_cos_sim_loss = (1 - F.cosine_similarity(sal_from_mr, gt_norm)).sum() * self.mr_to_hd_loss_coef

        hard_loss = torch.zeros(1).to(saliency_scores.device)
        if self.hard_pos_neg_loss:
            hard_loss = (self.adaptive_hard_neg_loss_saliency(sal_norm, gt_norm, targets, indices, epoch_i)
                         + self.adaptive_hard_pos_loss_saliency(sal_norm, gt_norm, targets, indices, epoch_i))

        # print(loss_saliency, loss_rank_contrastive)
        # loss_saliency = loss_saliency + loss_rank_contrastive

        loss_saliency = (loss_saliency + loss_rank_contrastive + loss_neg_pair
                         + cos_sim_err.sum()
                         + sal_from_mr_cos_sim_loss.sum()
                         + hard_loss.sum()
                         )
        # loss_saliency = loss_rank_contrastive
        return {"loss_saliency": loss_saliency}

    def loss_contrastive_align(self, outputs, targets, indices, log=True, epoch_i=0):
        """encourage higher scores between matched query span and input text"""
        normalized_text_embed = outputs["proj_txt_mem"]  # (bsz, #tokens, d)  text tokens
        normalized_img_embed = outputs["proj_queries"]  # (bsz, #queries, d)
        logits = torch.einsum(
            "bmd,bnd->bmn", normalized_img_embed, normalized_text_embed)  # (bsz, #queries, #tokens)
        logits = logits.sum(2) / self.temperature  # (bsz, #queries)
        idx = self._get_src_permutation_idx(indices)
        positive_map = torch.zeros_like(logits, dtype=torch.bool)
        positive_map[idx] = True
        positive_logits = logits.masked_fill(~positive_map, 0)

        pos_term = positive_logits.sum(1)  # (bsz, )
        num_pos = positive_map.sum(1)  # (bsz, )
        neg_term = logits.logsumexp(1)  # (bsz, )
        loss_nce = - pos_term / num_pos + neg_term  # (bsz, )
        losses = {"loss_contrastive_align": loss_nce.mean()
                                            + self.loss_align_vid_txt(outputs, targets, indices)}
        return losses

    def loss_align_vid_txt(self, outputs, targets, indices, log=True, epoch_i=0):
        """Encourage higher scores between matched video and input text at a clip-sentence level."""
        # Extracting video and text embeddings
        if "src_txt" not in outputs or "src_vid" not in outputs:
            return torch.zeros(1).to(targets["saliency_all_labels"].device)
        normalized_text_embed = outputs["src_txt"]  # (bsz, #tokens, d) - text tokens
        normalized_img_embed = outputs["src_vid"]  # (bsz, #clips, d) - video tokens

        # Pooling to get sentence-level text embeddings
        # Assuming mean pooling for simplicity; can replace with other methods if needed
        sentence_level_text_embed = normalized_text_embed.mean(dim=1)  # (bsz, d)

        # Calculating cosine similarity between sentence-level text embeddings and clip-level video embeddings
        saliency_scores = F.cosine_similarity(
            sentence_level_text_embed.unsqueeze(1),  # (bsz, 1, d)
            normalized_img_embed,  # (bsz, #clips, d)
            dim=-1  # cosine similarity along the feature dimension
        )  # (bsz, #clips)

        # Normalizing saliency scores for comparison with ground truth
        gt_tensor = targets["saliency_all_labels"]  # Ground truth saliency scores
        saliency_scores = (saliency_scores - saliency_scores.min()) / (gt_tensor.max() - gt_tensor.min())

        # Normalizing both ground truth and predicted saliency scores
        gt_norm = F.normalize(gt_tensor, dim=1, p=2)  # (bsz, #clips)
        sal_norm = F.normalize(saliency_scores, dim=1, p=2)  # (bsz, #clips)

        # Calculating cosine similarity loss
        cos_sim = F.cosine_similarity(gt_norm, sal_norm, dim=1)  # (bsz,)
        cos_sim_err = (1 - cos_sim)  # (bsz,)

        # Returning the summed loss
        return cos_sim_err.sum()

    def loss_contrastive_align_vid_txt_back(self, outputs, targets, indices, log=True, epoch_i=0):
        """encourage higher scores between matched video and input text"""
        # Assuming normalized_text_embed and normalized_img_embed are from outputs
        normalized_text_embed = outputs["proj_txt_mem"]  # (bsz, #tokens, d)  text tokens
        normalized_img_embed = outputs["proj_vid_mem"]  # (bsz, #clips, d)   video tokens
        # src_txt = outputs["src_txt"]  # (bsz, #clips, d)   video tokens

        bs, vl, dim = normalized_img_embed.shape
        _, ql, _ = normalized_text_embed.shape

        # softmax = nn.Softmax(dim=0)
        # txt_attention_weights = softmax(normalized_text_embed)  # Shape: (10, 1)

        mat_v = torch.ones(vl, dim).to(normalized_img_embed.device)
        mat_q = torch.ones(ql, dim).to(normalized_text_embed.device)

        d_v = torch.matmul(normalized_img_embed.mean(dim=2), mat_v)
        d_q = torch.matmul(normalized_text_embed.mean(dim=2), mat_q)

        logits = torch.matmul(d_v, d_q.T) * torch.exp(torch.tensor(self.temperature).to(d_v.device))

        # symmetric loss function
        labels = torch.arange(bs).to(logits.device)
        loss_v = F.cross_entropy(logits, labels)
        loss_q = F.cross_entropy(logits.T, labels)
        loss = (loss_v + loss_q) / 2

        losses = {"loss_contrastive_align": loss}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx  # two 1D tensors of the same length

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, epoch_i=0, **kwargs):
        loss_map = {
            "spans": self.loss_spans,
            "labels": self.loss_labels,
            "contrastive_align": self.loss_contrastive_align,
            "saliency": self.loss_saliency,
            "iou": self.loss_iou,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, epoch_i=epoch_i, **kwargs)

    def forward(self, outputs, targets, epoch_i=0):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # list(tuples), each tuple is (pred_span_indices, tgt_span_indices)

        # only for HL, do not use matcher
        if self.use_matcher:
            indices = self.matcher(outputs_without_aux, targets)
            losses_target = self.losses
        else:
            indices = None
            losses_target = ["saliency"]

        # Compute all the requested losses
        losses = {}
        # for loss in self.losses:
        for loss in losses_target:
            losses.update(self.get_loss(loss, outputs, targets, indices, epoch_i=epoch_i))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # indices = self.matcher(aux_outputs, targets)
                if self.use_matcher:
                    indices = self.matcher(aux_outputs, targets)
                    losses_target = self.losses
                else:
                    indices = None
                    losses_target = ["saliency"]
                    # for loss in self.losses:
                for loss in losses_target:
                    if "saliency" == loss:  # skip as it is only in the top layer
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, epoch_i=epoch_i, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


def build_model(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/qd_detr/issues/108#issuecomment-650269223
    device = torch.device(args.device)

    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    model = VideoLight(
        transformer,
        position_embedding,
        txt_position_embedding,
        txt_dim=args.t_feat_dim,
        vid_dim=args.v_feat_dim,
        aud_dim=args.a_feat_dim,
        num_queries=args.num_queries,
        input_dropout=args.input_dropout,
        aux_loss=args.aux_loss,
        mr_to_hd_loss=args.mr_to_hd_loss,
        contrastive_align_loss=args.contrastive_align_loss,
        contrastive_hdim=args.contrastive_hdim,
        span_loss_type=args.span_loss_type,
        use_txt_pos=args.use_txt_pos,
        n_input_proj=args.n_input_proj,
        clip_len=args.clip_length,
        fra=not getattr(args, 'no_feature_refinement', False),
    )

    matcher = build_matcher(args)
    weight_dict = {"loss_span": args.span_loss_coef,
                   "loss_giou": args.giou_loss_coef,
                   "loss_label": args.label_loss_coef,
                   "loss_saliency": args.lw_saliency,
                   "loss_iou": getattr(args, 'iou_loss_coef', 1.0)}
    if args.contrastive_align_loss:
        weight_dict["loss_contrastive_align"] = args.contrastive_align_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items() if k != "loss_saliency"})
        weight_dict.update(aux_weight_dict)

    losses = ['spans', 'labels', 'saliency']
    if getattr(args, 'use_iou_head', False):
        losses.append('iou')
    if args.contrastive_align_loss:
        losses += ["contrastive_align"]

    # For tvsum dataset
    use_matcher = not (args.dset_name == 'tvsum' or args.dset_name == 'youtube_uni')

    criterion = SetCriterion(
        matcher=matcher, weight_dict=weight_dict, losses=losses,
        eos_coef=args.eos_coef, temperature=args.temperature,
        span_loss_type=args.span_loss_type, max_v_l=args.max_v_l,
        saliency_margin=args.saliency_margin, use_matcher=use_matcher,
        n_epoch=args.n_epoch, hard_pos_neg_loss=args.hard_pos_neg_loss,
        hard_pos_neg_loss_coef=args.hard_pos_neg_loss_coef, clip_len=args.clip_length,
        mr_to_hd_loss=args.mr_to_hd_loss, mr_to_hd_loss_coef=args.mr_to_hd_loss_coef,
        cos_sim_loss_coef=args.cos_sim_loss_coef,
    )
    criterion.to(device)
    return model, criterion
