# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math
import numpy as np
from .attention import MultiheadAttention
from .components import Conv1D


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    center_embed = pos_tensor[:, :, 0] * scale
    pos_x = center_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)

    span_embed = pos_tensor[:, :, 1] * scale
    pos_w = span_embed[:, :, None] / dim_t
    pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

    pos = torch.cat((pos_x, pos_w), dim=2)
    return pos


class VLTransformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_queries=2, num_cmi_layers=2, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, query_dim=2,
                 keep_query_pos=False, query_scale_type='cond_elewise',
                 num_patterns=0,
                 modulate_t_attn=True,
                 bbox_embed_diff_each_layer=False,
                 hd_to_mr_gate=False,
                 cmi_gate=False,
                 decoder_sent_gate=False,
                 span_gaussian_mask=False,
                 task_decouple_gate=False,
                 ):
        super().__init__()

        self.hd_to_mr_gate = hd_to_mr_gate
        self.decoder_sent_gate = decoder_sent_gate
        self.task_decouple_gate = task_decouple_gate
        if hd_to_mr_gate:
            self.saliency_gate_proj = nn.Linear(d_model, d_model)
        if task_decouple_gate:
            self.mr_gate_proj = nn.Linear(d_model, d_model)
            self.hd_gate_proj = nn.Linear(d_model, d_model)

        # t2v_encoder_layer = T2V_TransformerEncoderLayer(d_model, nhead, dim_feedforward,
        #                                         dropout, activation, normalize_before)
        t2v_encoder_layer = CMILayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,
                                                cmi_gate=cmi_gate)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.t2v_encoder = TransformerEncoder(t2v_encoder_layer, num_cmi_layers, encoder_norm)


        # TransformerEncoderLayerThin
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # TransformerDecoderLayerThin
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, keep_query_pos=keep_query_pos)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=d_model, query_dim=query_dim, keep_query_pos=keep_query_pos, query_scale_type=query_scale_type,
                                          modulate_t_attn=modulate_t_attn,
                                          bbox_embed_diff_each_layer=bbox_embed_diff_each_layer,
                                          decoder_sent_gate=decoder_sent_gate,
                                          span_gaussian_mask=span_gaussian_mask)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries
        self.num_patterns = num_patterns

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # for tvsum, add video_length in argument
    def forward(self, src, mask, query_embed, pos_embed, video_length=None, meta=None):
        """
        Args:
            src: (batch_size, L, d)
            mask: (batch_size, L)
            query_embed: (#queries, d)
            pos_embed: (batch_size, L, d) the same as src

        Returns:

        """
        # flatten NxCxHxW to HWxNxC
        global src_txt
        bs, l, d = src.shape
        src = src.permute(1, 0, 2)  # (L, batch_size, d)
        pos_embed = pos_embed.permute(1, 0, 2)   # (L, batch_size, d)
        refpoint_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # (#queries, batch_size, d)

        src = self.t2v_encoder(src, src_key_padding_mask=mask, pos=pos_embed, video_length=video_length)  # (L, batch_size, d)
        # print('after encoder : ',src.shape)

        # if meta is not None:
        #     src_txt = src[video_length + 1: ]
        #     cp_src = src.permute(1, 0, 2)
        #     src_txt = src_txt.permute(1, 0, 2)
        #
        #     for i in range(bs):
        #         qid = meta[i]['qid']
        #         string =  meta[i]['query']
        #         words = string.split()
        #         word_count = len(words)
        #         np.savez_compressed(f'plot/qvhl/out_cmi/{qid}.npz', vid=cp_src[i][1:video_length + 1].cpu().detach().numpy(),
        #                             txt=src_txt[i][: word_count + 2].cpu().detach().numpy())

        # Extract sentence-level text feature for decoder gated fusion (before stripping text)
        src_txt_global = None
        if self.decoder_sent_gate:
            txt_feats = src[video_length + 1:]  # (L_txt, bs, d) - text tokens after t2v_encoder
            src_txt_global = txt_feats.max(dim=0)[0].unsqueeze(0)  # (1, bs, d)

        src = src[:video_length + 1]
        mask = mask[:, :video_length + 1]
        pos_embed = pos_embed[:video_length + 1]


        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)  # (L, batch_size, d)
        memory_global, memory_local = memory[0], memory[1:]
        mask_local = mask[:, 1:]
        pos_embed_local = pos_embed[1:]

        # HD→MR saliency gating: use global-token-aware saliency to gate memory before decoder
        if self.hd_to_mr_gate:
            memory_local_t = memory_local.transpose(0, 1)  # (batch_size, L_vid, d)
            # memory_global carries query-aware global context from encoder
            sal_scores = torch.sum(
                self.saliency_gate_proj(memory_local_t) * memory_global.unsqueeze(1),
                dim=-1
            ) / np.sqrt(self.d_model)  # (batch_size, L_vid)
            sal_gate = torch.sigmoid(sal_scores).unsqueeze(-1)  # (batch_size, L_vid, 1)
            memory_local = (memory_local_t * sal_gate + memory_local_t).transpose(0, 1)  # residual gating

        # if meta is not None:
        #     cp_src = memory_local.permute(1, 0, 2)
        #     for i in range(bs):
        #         qid = meta[i]['qid']
        #         string =  meta[i]['query']
        #         words = string.split()
        #         word_count = len(words)
        #         np.savez_compressed(f'plot/qvhl/out_final/{qid}.npz', vid=cp_src[i].cpu().detach().numpy(),
        #                             txt=src_txt[i][: word_count + 2].cpu().detach().numpy())

        tgt = torch.zeros(refpoint_embed.shape[0], bs, d).cuda()
        # Task-decoupled memory gating: separate gates for MR (decoder) and HD (saliency)
        if self.task_decouple_gate:
            memory_local_t = memory_local.transpose(0, 1)  # (bs, L_vid, d)
            mr_gate = torch.sigmoid(self.mr_gate_proj(memory_local_t))  # (bs, L_vid, d)
            hd_gate = torch.sigmoid(self.hd_gate_proj(memory_local_t))  # (bs, L_vid, d)
            mr_memory = (mr_gate * memory_local_t).transpose(0, 1)  # (L_vid, bs, d) for decoder
            hd_memory_out = hd_gate * memory_local_t  # (bs, L_vid, d) for saliency
            hs, references = self.decoder(tgt, mr_memory, memory_key_padding_mask=mask_local,
                              pos=pos_embed_local, refpoints_unsigmoid=refpoint_embed,
                              src_txt_global=src_txt_global)
            memory_local = hd_memory_out  # (batch_size, L, d)
        else:
            hs, references = self.decoder(tgt, memory_local, memory_key_padding_mask=mask_local,
                              pos=pos_embed_local, refpoints_unsigmoid=refpoint_embed,
                              src_txt_global=src_txt_global)  # (#layers, #queries, batch_size, d)
            memory_local = memory_local.transpose(0, 1)  # (batch_size, L, d)
        # hs = hs.transpose(1, 2)  # (#layers, batch_size, #qeries, d)
        # memory = memory.permute(1, 2, 0)  # (batch_size, d, L)
        return hs, references, memory_local, memory_global


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    # for tvsum, add kwargs
    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                **kwargs):
        output = src

        intermediate = []

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos, **kwargs)
            if self.return_intermediate:
                intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False,
                 d_model=256, query_dim=2, keep_query_pos=False, query_scale_type='cond_elewise',
                 modulate_t_attn=False,
                 bbox_embed_diff_each_layer=False,
                 decoder_sent_gate=False,
                 span_gaussian_mask=False,
                 ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate
        self.query_dim = query_dim

        # EaTR-style Gated Fusion: inject sentence text into queries (first layer only)
        self.decoder_sent_gate = decoder_sent_gate
        if decoder_sent_gate:
            self.gate_cross_attn = nn.MultiheadAttention(d_model, 1, dropout=0.1)
            self.gate_self_attn = nn.MultiheadAttention(d_model, 1, dropout=0.1)
            self.gate_linear = nn.Linear(d_model, d_model)
            self.gate_norm = nn.LayerNorm(d_model)
            self.gate_dropout = nn.Dropout(0.1)

        # Span-Anchor Gaussian Attention Mask
        self.span_gaussian_mask = span_gaussian_mask
        if span_gaussian_mask:
            self.gaussian_alpha = nn.Parameter(torch.tensor(5.0))

        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']
        self.query_scale_type = query_scale_type
        if query_scale_type == 'cond_elewise':
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        elif query_scale_type == 'cond_scalar':
            self.query_scale = MLP(d_model, d_model, 1, 2)
        elif query_scale_type == 'fix_elewise':
            self.query_scale = nn.Embedding(num_layers, d_model)
        else:
            raise NotImplementedError("Unknown query_scale_type: {}".format(query_scale_type))

        self.ref_point_head = MLP(d_model, d_model, d_model, 2)

        # self.bbox_embed = None
        # for DAB-deter
        if bbox_embed_diff_each_layer:
            self.bbox_embed = nn.ModuleList([MLP(d_model, d_model, 2, 3) for i in range(num_layers)])
        else:
            self.bbox_embed = MLP(d_model, d_model, 2, 3)
        # init bbox_embed
        if bbox_embed_diff_each_layer:
            for bbox_embed in self.bbox_embed:
                nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
                nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        self.d_model = d_model
        self.modulate_t_attn = modulate_t_attn
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer

        if modulate_t_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 1, 2)

        if not keep_query_pos:
            for layer_id in range(num_layers - 1):
                self.layers[layer_id + 1].ca_qpos_proj = None

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
                src_txt_global: Optional[Tensor] = None,  # (1, bs, d) sentence feature
                ):
        output = tgt

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        hw, bs, _ = memory.shape  # L_vid, batch_size, d_model

        for layer_id, layer in enumerate(self.layers):
            obj_center = reference_points[..., :self.query_dim]
            # get sine embedding for the query vector
            query_sine_embed = gen_sineembed_for_position(obj_center)
            query_pos = self.ref_point_head(query_sine_embed)
            # For the first decoder layer, we do not apply transformation over p_s
            if self.query_scale_type != 'fix_elewise':
                if layer_id == 0:
                    pos_transformation = 1
                else:
                    pos_transformation = self.query_scale(output)
            else:
                pos_transformation = self.query_scale.weight[layer_id]

            # apply transformation
            query_sine_embed = query_sine_embed * pos_transformation

            # modulated HW attentions
            if self.modulate_t_attn:
                reft_cond = self.ref_anchor_head(output).sigmoid()  # nq, bs, 1
                query_sine_embed *= (reft_cond[..., 0] / obj_center[..., 1]).unsqueeze(-1)

            # EaTR-style Gated Fusion: inject sentence text into queries (first layer only)
            if self.decoder_sent_gate and layer_id == 0 and src_txt_global is not None:
                gf = self.gate_cross_attn(query=output, key=src_txt_global, value=src_txt_global)[0]
                gate = (output * gf).sigmoid()
                gf_sum = output + gf
                gf_out = self.gate_self_attn(query=gf_sum, key=gf_sum, value=gf_sum)[0]
                output = self.gate_dropout(F.relu(self.gate_linear(gate * gf_out))) + output
                output = self.gate_norm(output)

            # Span-Anchor Gaussian Attention Mask: focus cross-attention on predicted span region
            layer_memory_mask = memory_mask
            if self.span_gaussian_mask:
                nq = reference_points.shape[0]
                positions = torch.arange(hw, device=reference_points.device).float() / hw  # (L_vid,)
                center = reference_points[..., 0]  # (nq, bs)
                width = reference_points[..., 1]   # (nq, bs)
                sigma = (width / 2).clamp(min=0.05)  # (nq, bs)
                gaussian = torch.exp(-((positions[None, None, :] - center[:, :, None]) ** 2) / (2 * sigma[:, :, None] ** 2))  # (nq, bs, hw)
                # Additive mask: suppress attention outside predicted span
                gauss_bias = -self.gaussian_alpha * (1.0 - gaussian)  # (nq, bs, hw)
                nhead = layer.nhead
                gauss_bias = gauss_bias.permute(1, 0, 2)  # (bs, nq, hw)
                gauss_bias = gauss_bias.unsqueeze(1).expand(-1, nhead, -1, -1).contiguous()
                layer_memory_mask = gauss_bias.reshape(bs * nhead, nq, hw)  # (bs*nhead, nq, hw)

            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=layer_memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed,
                           is_first=(layer_id == 0))

            # iter update
            if self.bbox_embed is not None:
                if self.bbox_embed_diff_each_layer:
                    tmp = self.bbox_embed[layer_id](output)
                else:
                    tmp = self.bbox_embed(output)
                # import ipdb; ipdb.set_trace()
                tmp[..., :self.query_dim] += inverse_sigmoid(reference_points)
                new_reference_points = tmp[..., :self.query_dim].sigmoid()
                if layer_id != self.num_layers - 1:
                    ref_points.append(new_reference_points)
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if self.bbox_embed is not None:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    torch.stack(ref_points).transpose(1, 2),
                ]
            else:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    reference_points.unsqueeze(0).transpose(1, 2)
                ]

        return output.unsqueeze(0)


class TransformerEncoderLayerThin(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        # self.linear1 = nn.Linear(d_model, dim_feedforward)
        # self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src2 = self.linear(src2)
        src = src + self.dropout(src2)
        src = self.norm(src)
        # src = src + self.dropout1(src2)
        # src = self.norm1(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # src = src + self.dropout2(src2)
        # src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        """not used"""
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class CMILayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, cmi_gate=False):
        super().__init__()
        self.cross_attn_1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_3 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.linear3 = nn.Linear(d_model, d_model)
        self.linear4 = nn.Linear(d_model, dim_feedforward)
        self.linear5 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.nhead = nhead

        self.cmi_gate = cmi_gate
        if cmi_gate:
            self.cross_gate = nn.Linear(d_model, d_model)


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     video_length=None):
        assert video_length is not None

        # print('before src shape :', src.shape)
        pos_src = self.with_pos_embed(src, pos)

        global_token, video, text = (src[0].unsqueeze(0), src[1:video_length + 1], src[video_length + 1:])
        pos_vid, pos_txt = pos_src[1:video_length + 1], pos_src[video_length + 1:]

        qmask_vid, kmask_vid = (src_key_padding_mask[:, 1:video_length + 1].unsqueeze(2),
                                src_key_padding_mask[:, video_length + 1:].unsqueeze(1))
        attn_mask_vid = torch.matmul(qmask_vid.float(), kmask_vid.float()).bool().repeat(self.nhead, 1, 1)

        qmask_txt, kmask_txt = (src_key_padding_mask[:, video_length + 1:].unsqueeze(2),
                                src_key_padding_mask[:, 1:video_length + 1].unsqueeze(1))
        attn_mask_txt = torch.matmul(qmask_txt.float(), kmask_txt.float()).bool().repeat(self.nhead, 1, 1)

        # print(attn_mask.shape)
        # print(attn_mask[0][0])
        # print(q.shape) 75 32 256
        # print(k.shape) 26 32 256

        q, k, v = pos_vid, pos_txt, text
        v_out = self.cross_attn_1(query=q, key=k, value=v, attn_mask=attn_mask_vid,
                                  key_padding_mask=src_key_padding_mask[:, video_length + 1:])[0]
        if self.cmi_gate:
            # Gate controls per-clip how much text info flows in; irrelevant clips → gate≈0
            gate = torch.sigmoid(self.cross_gate(video))  # (L_vid, batch_size, d)
            v_out = self.norm1(video + gate * self.linear1(self.dropout(v_out)))
        else:
            v_out = self.norm1(video + self.linear1(self.dropout(v_out)))
        # v_out = video + self.linear1(self.dropout(v_out))

        q, k, v = pos_txt, pos_vid, video
        t_out = self.cross_attn_2(query=q, key=k, value=v, attn_mask=attn_mask_txt,
                                  key_padding_mask=src_key_padding_mask[:, 1:video_length + 1])[0]
        # t_out = self.norm2(text + self.linear2(self.dropout(t_out)))
        t_out = text + self.linear2(self.dropout(t_out))

        c_out = self.cross_attn_3(query=v_out, key=t_out, value=t_out, attn_mask=attn_mask_vid,
                                  key_padding_mask=src_key_padding_mask[:, video_length + 1:])[0]
        c_out = self.norm3(v_out + self.linear3(self.dropout(c_out)))

        # v_out = self.self_attn(query=c_out, key=c_out, value=c_out, attn_mask=attn_mask_vid,
        #                        key_padding_mask=src_key_padding_mask[:, 1:video_length + 1])[0]
        # v_out = self.norm4(c_out + self.dropout(v_out))

        v_out = self.linear5(self.dropout(self.activation(self.linear4(c_out))))
        v_out = self.norm4(v_out + self.dropout(v_out))


        src2 = torch.cat([global_token, v_out], dim=0)
        src = torch.cat([src2, src[video_length + 1:]], dim=0)
        # print('after src shape :',src.shape)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                     video_length=None):
        print('before src shape :', src.shape)
        src = self.norm4(src)
        # print('before src shape :', src.shape)
        pos_src = self.with_pos_embed(src, pos)

        global_token, video, text = (src[0].unsqueeze(0), src[1:video_length + 1], src[video_length + 1:])
        pos_vid, pos_txt = pos_src[1:video_length + 1], pos_src[video_length + 1:]

        qmask_vid, kmask_vid = (src_key_padding_mask[:, 1:video_length + 1].unsqueeze(2),
                                src_key_padding_mask[:, video_length + 1:].unsqueeze(1))
        attn_mask_vid = torch.matmul(qmask_vid.float(), kmask_vid.float()).bool().repeat(self.nhead, 1, 1)

        qmask_txt, kmask_txt = (src_key_padding_mask[:, video_length + 1:].unsqueeze(2),
                                src_key_padding_mask[:, 1:video_length + 1].unsqueeze(1))
        attn_mask_txt = torch.matmul(qmask_txt.float(), kmask_txt.float()).bool().repeat(self.nhead, 1, 1)

        # print(attn_mask.shape)
        # print(attn_mask[0][0])
        # print(q.shape) 75 32 256
        # print(k.shape) 26 32 256

        q, k, v = pos_vid, pos_txt, text
        v_out = self.cross_attn_1(query=q, key=k, value=v, attn_mask=attn_mask_vid,
                                  key_padding_mask=src_key_padding_mask[:, video_length + 1:])[0]
        v_out = self.norm1(video + self.linear1(self.dropout(v_out)))

        q, k, v = pos_txt, pos_vid, video
        t_out = self.cross_attn_2(query=q, key=k, value=v, attn_mask=attn_mask_txt,
                                  key_padding_mask=src_key_padding_mask[:, 1:video_length + 1])[0]
        t_out = self.norm2(text + self.linear2(self.dropout(t_out)))

        c_out = self.cross_attn_3(query=v_out, key=t_out, value=t_out, attn_mask=attn_mask_vid,
                                  key_padding_mask=src_key_padding_mask[:, video_length + 1:])[0]
        c_out = self.norm3(v_out + self.linear3(self.dropout(c_out)))

        # v_out = self.self_attn(query=c_out, key=c_out, value=c_out, attn_mask=attn_mask_vid,
        #                        key_padding_mask=src_key_padding_mask[:, 1:video_length + 1])[0]
        # v_out = self.norm4(c_out + self.dropout4(v_out))

        v_out = self.linear5(self.dropout(self.activation(self.linear4(c_out))))
        v_out = v_out + self.dropout(v_out)

        src2 = torch.cat([global_token, v_out], dim=0)
        src = torch.cat([src2, src[video_length + 1:]], dim=0)
        # print('after src shape :',src.shape)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                **kwargs):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        # For tvsum, add kwargs
        return self.forward_post(src, src_mask, src_key_padding_mask, pos, **kwargs)


class T2V_TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.nhead = nhead

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     video_length=None):
        
        assert video_length is not None
        
        # print('before src shape :', src.shape)
        pos_src = self.with_pos_embed(src, pos)
        global_token, q, k, v = src[0].unsqueeze(0), pos_src[1:video_length + 1], pos_src[video_length + 1:], src[video_length + 1:]

        # print(src_key_padding_mask.shape) # torch.Size([32, 102])
        # print(src_key_padding_mask[:, 1:76].permute(1,0).shape) # torch.Size([75, 32])
        # print(src_key_padding_mask[:, 76:].shape) # torch.Size([32, 26])

        qmask, kmask = src_key_padding_mask[:, 1:video_length + 1].unsqueeze(2), src_key_padding_mask[:, video_length + 1:].unsqueeze(1)
        attn_mask = torch.matmul(qmask.float(), kmask.float()).bool().repeat(self.nhead, 1, 1)
        # print(attn_mask.shape)
        # print(attn_mask[0][0])
        # print(q.shape) 75 32 256
        # print(k.shape) 26 32 256


        src2 = self.self_attn(q, k, value=v, attn_mask=attn_mask,
                              key_padding_mask=src_key_padding_mask[:, video_length + 1:])[0]
        src2 = src[1:video_length + 1] + self.dropout1(src2)
        src3 = self.norm1(src2)
        src3 = self.linear2(self.dropout(self.activation(self.linear1(src3))))
        src2 = src2 + self.dropout2(src3)
        src2 = self.norm2(src2)
        src2 = torch.cat([global_token, src2], dim=0)
        src = torch.cat([src2, src[video_length + 1:]])
        # print('after src shape :',src.shape)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        print('before src shape :', src.shape)
        src2 = self.norm1(src)
        pos_src = self.with_pos_embed(src2, pos)
        global_token, q, k, v = src[0].unsqueeze(0), pos_src[1:76], pos_src[76:], src2[76:]
        # print(q.shape) # 100 32 256


        src2 = self.self_attn(q, k, value=v, attn_mask=src_key_padding_mask[:, 1:76].permute(1,0),
                              key_padding_mask=src_key_padding_mask[:, 76:])[0]
        src2 = src[1:76] + self.dropout1(src2)
        src3 = self.norm1(src2)
        src3 = self.linear2(self.dropout(self.activation(self.linear1(src3))))
        src2 = src2 + self.dropout2(src3)
        src2 = self.norm2(src2)
        src2 = torch.cat([global_token, src2], dim=0)
        src = torch.cat([src2, src[76:]])
        print('after src shape :',src.shape)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                **kwargs):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        # For tvsum, add kwargs
        return self.forward_post(src, src_mask, src_key_padding_mask, pos, **kwargs)

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, keep_query_pos=False,
                 rm_self_attn_decoder=False):
        super().__init__()
        # Decoder Self-Attention
        if not rm_self_attn_decoder:
            self.sa_qcontent_proj = nn.Linear(d_model, d_model)
            self.sa_qpos_proj = nn.Linear(d_model, d_model)
            self.sa_kcontent_proj = nn.Linear(d_model, d_model)
            self.sa_kpos_proj = nn.Linear(d_model, d_model)
            self.sa_v_proj = nn.Linear(d_model, d_model)
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead
        self.rm_self_attn_decoder = rm_self_attn_decoder

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.keep_query_pos = keep_query_pos

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_sine_embed=None,
                is_first=False):

        # ========== Begin of Self-Attention =============
        if not self.rm_self_attn_decoder:
            # Apply projections here
            # shape: num_queries x batch_size x 256
            q_content = self.sa_qcontent_proj(tgt)  # target is the input of the first decoder layer. zero by default.
            q_pos = self.sa_qpos_proj(query_pos)
            k_content = self.sa_kcontent_proj(tgt)
            k_pos = self.sa_kpos_proj(query_pos)
            v = self.sa_v_proj(tgt)

            num_queries, bs, n_model = q_content.shape
            hw, _, _ = k_content.shape

            q = q_content + q_pos
            k = k_content + k_pos

            tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            # ========== End of Self-Attention =============

            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos)

        # For the first decoder layer, we concatenate the positional embedding predicted from
        # the object query (the positional embedding) into the original query (key) in DETR.
        if is_first or self.keep_query_pos:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(num_queries, bs, self.nhead, n_model // self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model // self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        k = k.view(hw, bs, self.nhead, n_model // self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model // self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        tgt2 = self.cross_attn(query=q,
                               key=k,
                               value=v, attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)[0]
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TransformerDecoderLayerThin(nn.Module):
    """removed intermediate layer"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, d_model)
        # self.linear1 = nn.Linear(d_model, dim_feedforward)
        # self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # self.dropout3 = nn.Dropout(dropout)

        # self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt2 = self.linear1(tgt2)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # tgt = tgt + self.dropout2(tgt2)
        # tgt = self.norm2(tgt)
        # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        # tgt = tgt + self.dropout3(tgt2)
        # tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# def build_transformer(args):
#     return Transformer(
#         d_model=args.hidden_dim,
#         dropout=args.dropout,
#         nhead=args.nheads,
#         dim_feedforward=args.dim_feedforward,
#         num_encoder_layers=args.enc_layers,
#         num_decoder_layers=args.dec_layers,
#         normalize_before=args.pre_norm,
#         return_intermediate_dec=True,
#     )

def build_transformer(args):
    return VLTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_cmi_layers=args.cmi_layers,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        activation='prelu',
        hd_to_mr_gate=getattr(args, 'hd_to_mr_gate', False),
        cmi_gate=getattr(args, 'cmi_gate', False),
        decoder_sent_gate=getattr(args, 'decoder_sent_gate', False),
        span_gaussian_mask=getattr(args, 'span_gaussian_mask', False),
        task_decouple_gate=getattr(args, 'task_decouple_gate', False),
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "mish":
        return nn.Mish()
    if activation == "selu":
        return F.selu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
