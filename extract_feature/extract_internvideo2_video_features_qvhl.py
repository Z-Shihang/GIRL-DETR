"""
Extract InternVideo2-L/14 video features for QVHighlights dataset.
Outputs .npz files with key "features" of shape (T, 768) float16.

The model is InternVideo2-distill-L/14 (embed_dim=1024, depth=24).
We extract the attention-pooled CLIP-aligned feature (768-dim) per clip of 8 frames.
Videos are sampled at 1 fps (matching QVHighlights existing features), then grouped
into sliding windows of 8 frames (stride=8, no overlap) to produce per-second features.

Requirements: torch, timm, einops, ffmpeg-python, numpy, tqdm

Usage:
    conda run -n video_lights python extract_feature/extract_internvideo2_video_features_qvhl.py --gpu 0
"""

import math
import logging
import numpy as np
import os
from os.path import join
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from functools import partial
from tqdm import tqdm
from PIL import Image
from einops import rearrange

import ffmpeg

# ============================================================
# Minimal InternVideo2 model code (no flash_attn dependency)
# Adapted from InternVideo2/multi_modality/models/backbones/internvideo2/internvideo2.py
# ============================================================

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

logger = logging.getLogger(__name__)


def get_3d_sincos_pos_embed(embed_dim, grid_size, t_size, cls_token=False):
    """3D sin-cos position embedding. grid_size: int of spatial, t_size: int of temporal."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])

    grid_t = np.arange(t_size, dtype=np.float32).reshape([1, t_size, 1, 1])

    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid, grid_t)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid, grid_t):
    assert embed_dim % 4 == 0
    embed_dim_spatial = embed_dim // 4 * 3
    embed_dim_temporal = embed_dim // 4

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim_spatial // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim_spatial // 2, grid[1])
    emb_t = get_1d_sincos_pos_embed_from_grid(embed_dim_temporal, grid_t)

    T = grid_t.shape[1]
    H = grid.shape[2]
    W = grid.shape[3]

    emb_h = np.tile(emb_h, (T, 1))
    emb_w = np.tile(emb_w, (T, 1))
    emb_t = np.repeat(emb_t, H * W, axis=0)

    emb = np.concatenate([emb_t, emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False, force_fp32=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
        self.force_fp32 = force_fp32

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        if self.force_fp32:
            output_type = x.dtype
            out = x.float().mul_(self.gamma.float()) if self.inplace else x.float() * self.gamma.float()
            return out.to(dtype=output_type)
        else:
            out = x.mul_(self.gamma) if self.inplace else x * self.gamma
            return out


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 norm_layer=nn.LayerNorm, qk_normalization=False, **kwargs):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qk_normalization = qk_normalization
        self.q_norm = norm_layer(dim) if qk_normalization else nn.Identity()
        self.k_norm = norm_layer(dim) if qk_normalization else nn.Identity()

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)

        attn = ((q * self.scale) @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0.,
                 attn_drop=0., init_values=None, drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, with_cp=False, qk_normalization=False,
                 layerscale_no_force_fp32=False, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop,
                              norm_layer=norm_layer, qk_normalization=qk_normalization)
        self.ls1 = LayerScale(dim, init_values=init_values,
                              force_fp32=(not layerscale_no_force_fp32)) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values,
                              force_fp32=(not layerscale_no_force_fp32)) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.with_cp = with_cp

    def forward(self, x, residual=None):
        def _inner_forward(x, residual=None):
            assert residual is None
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            return x

        if self.with_cp:
            return checkpoint.checkpoint(_inner_forward, x, residual)
        else:
            return _inner_forward(x, residual=residual)


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., attn_head_dim=None, out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        assert all_head_dim == dim

        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.k_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, k=None, v=None):
        B, N, C = x.shape
        N_k = k.shape[1]
        N_v = v.shape[1]

        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = self.k_bias
            v_bias = self.v_bias

        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q = q.reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        k = F.linear(input=k, weight=self.k.weight, bias=k_bias)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        v = F.linear(input=v, weight=self.v.weight, bias=v_bias)
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentiveBlock(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 attn_head_dim=None, out_dim=None):
        super().__init__()
        self.norm1_q = norm_layer(dim)
        self.norm1_k = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim,
            out_dim=out_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x_q, x_kv, pos_q, pos_k, bool_masked_pos, rel_pos_bias=None):
        x_q = self.norm1_q(x_q + pos_q)
        x_k = self.norm1_k(x_kv + pos_k)
        x_v = self.norm1_v(x_kv)
        x = self.cross_attn(x_q, k=x_k, v=x_v)
        return x


class AttentionPoolingBlock(AttentiveBlock):
    def forward(self, x):
        x_q = x.mean(1, keepdim=True)
        x_kv, pos_q, pos_k = x, 0, 0
        x = super().forward(x_q, x_kv, pos_q, pos_k, bool_masked_pos=None)
        x = x.squeeze(1)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 num_frames=8, tubelet_size=1, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            num_frames // tubelet_size,
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1]
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.num_img_patches = self.grid_size[1] * self.grid_size[2]

        self.proj = nn.Conv3d(
            in_channels=in_chans, out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
            stride=(tubelet_size, patch_size[0], patch_size[1])
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(3).permute(0, 2, 3, 1)  # B x C x T x HW => B x T x HW x C
        x = self.norm(x)
        return x


class Linear_Decoder(nn.Module):
    def __init__(self, in_channels=1408, out_channels=3200,
                 norm_layer=nn.LayerNorm, clip_norm_type='l2'):
        super().__init__()
        self.clip_norm_type = clip_norm_type
        # checkpoint has head.0 (Linear), head.2 (Linear) with GELU in between
        self.head = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, out_channels),
        )
        self.norm = norm_layer(out_channels)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.norm(self.head(x))
        if self.clip_norm_type == 'l2':
            x = x / x.norm(dim=-1, keepdim=True)
        elif self.clip_norm_type == 'none':
            pass
        return x


class PretrainInternVideo2(nn.Module):
    """InternVideo2 Vision Transformer backbone (no flash_attn dependency)."""

    def __init__(
            self,
            in_chans=3, patch_size=14, img_size=224,
            qkv_bias=False, drop_path_rate=0.25,
            embed_dim=1408, num_heads=16, mlp_ratio=48/11,
            init_values=1e-5, qk_normalization=True,
            depth=40,
            attn_pool_num_heads=16, clip_embed_dim=768,
            layerscale_no_force_fp32=False,
            num_frames=8, tubelet_size=1,
            sep_pos_embed=False, sep_image_video_pos_embed=False,
            use_checkpoint=False, checkpoint_num=0,
            clip_teacher_embed_dim=3200,
            clip_teacher_final_dim=768,
            clip_norm_type='l2',
            clip_return_layer=1,
            clip_student_return_interval=1,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.clip_norm_type = clip_norm_type

        self.return_index = []
        for i in range(clip_return_layer):
            self.return_index.append(depth - int(i * clip_student_return_interval) - 1)

        norm_layer_for_blocks = partial(RMSNorm, eps=1e-6)
        self.norm_layer_for_blocks = norm_layer_for_blocks

        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim,
            num_frames=num_frames, tubelet_size=tubelet_size,
        )
        num_patches = self.patch_embed.num_patches
        num_img_patches = self.patch_embed.num_img_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.sep_pos_embed = sep_pos_embed
        self.sep_image_video_pos_embed = sep_image_video_pos_embed
        if sep_image_video_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            self.img_pos_embed = nn.Parameter(torch.zeros(1, num_img_patches + 1, embed_dim))
            self.clip_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            self.clip_img_pos_embed = nn.Parameter(torch.zeros(1, num_img_patches + 1, embed_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            self.clip_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        with_cp_list = [False] * depth
        if use_checkpoint:
            for idx in range(depth):
                if idx < checkpoint_num:
                    with_cp_list[idx] = True

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias,
                  norm_layer=norm_layer_for_blocks,
                  drop_path=dpr[i], init_values=init_values, attn_drop=0.,
                  with_cp=with_cp_list[i],
                  qk_normalization=qk_normalization,
                  layerscale_no_force_fp32=layerscale_no_force_fp32)
            for i in range(depth)])

        self.clip_projector = AttentionPoolingBlock(
            dim=embed_dim, num_heads=attn_pool_num_heads, qkv_bias=True,
            qk_scale=None, drop=0., attn_drop=0.,
            norm_layer=partial(nn.LayerNorm, eps=1e-5), out_dim=clip_embed_dim)

        self.clip_decoder = nn.ModuleList([
            Linear_Decoder(
                in_channels=embed_dim, out_channels=clip_teacher_embed_dim,
                norm_layer=partial(nn.LayerNorm, eps=1e-5),
                clip_norm_type=clip_norm_type
            ) for _ in range(clip_return_layer)
        ])
        self.final_clip_decoder = nn.Identity()
        if clip_teacher_final_dim > 0:
            self.final_clip_decoder = Linear_Decoder(
                in_channels=clip_embed_dim, out_channels=clip_teacher_final_dim,
                norm_layer=partial(nn.LayerNorm, eps=1e-5),
                clip_norm_type=clip_norm_type
            )

        self.init_pos_embed()
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def init_pos_embed(self):
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            self.patch_embed.grid_size[1],
            self.patch_embed.grid_size[0],
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.clip_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        if self.sep_image_video_pos_embed:
            img_pos_embed = get_3d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                self.patch_embed.grid_size[1],
                1, cls_token=True
            )
            self.img_pos_embed.data.copy_(torch.from_numpy(img_pos_embed).float().unsqueeze(0))
            self.clip_img_pos_embed.data.copy_(torch.from_numpy(img_pos_embed).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    @property
    def dtype(self):
        return self.patch_embed.proj.weight.dtype

    def forward(self, x, mask=None, use_image=False):
        """
        Args:
            x: [B, C, T, H, W]
            mask: optional bool mask
            use_image: if True, treat as single image
        Returns:
            x_vis: [B, N, embed_dim]
            x_pool_vis: [B, clip_embed_dim]
            x_clip_align: [K, B, N, clip_teacher_embed_dim]
            x_align: [B, clip_teacher_final_dim]
        """
        x = self.patch_embed(x.type(self.dtype))
        B, T, L, C = x.shape
        x = x.view([B, T * L, C])

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if use_image:
            if self.sep_image_video_pos_embed:
                pos_embed = self.img_pos_embed
            else:
                cls_pos_embed = self.pos_embed[:, 0:1, :]
                img_pos_embed = self.pos_embed[:, 1:, :].view(
                    1, self.num_frames, self.patch_embed.num_patches // self.num_frames, self.embed_dim
                ).mean(dim=1)
                pos_embed = torch.cat([cls_pos_embed, img_pos_embed], dim=1)
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed

        if mask is not None:
            x = x[~mask].reshape(B, -1, C)
        else:
            x = x.reshape(B, -1, C)

        residual = None
        x_clip = []
        for idx, blk in enumerate(self.blocks):
            if isinstance(x, tuple) and len(x) == 2:
                x, residual = x
            x = blk(x, residual=residual)
            if idx in self.return_index:
                if isinstance(x, tuple) and len(x) == 2:
                    tmp_x, tmp_residual = x
                    if residual is not None:
                        x_clip.append(tmp_x + tmp_residual)
                else:
                    x_clip.append(x)

        if isinstance(x, tuple) and len(x) == 2:
            x, residual = x
            if residual is not None:
                x = x + residual

        x_vis = x
        x_pool_vis = self.clip_projector(x_vis)
        x_align = self.final_clip_decoder(x_pool_vis)

        x_clip = torch.stack(x_clip)
        K, B_, _, C_CLIP = x_clip.shape
        if use_image:
            if self.sep_image_video_pos_embed:
                clip_pos_embed = self.clip_img_pos_embed
            else:
                clip_cls_pos_embed = self.clip_pos_embed[:, 0:1, :]
                clip_img_pos_embed = self.clip_pos_embed[:, 1:, :].view(
                    1, self.num_frames, self.patch_embed.num_patches // self.num_frames, self.embed_dim
                ).mean(dim=1)
                clip_pos_embed = torch.cat([clip_cls_pos_embed, clip_img_pos_embed], dim=1)
        else:
            clip_pos_embed = self.clip_pos_embed

        clip_pos_embed = clip_pos_embed.repeat(B, 1, 1)
        if mask is not None:
            x_clip = x_clip + clip_pos_embed[~mask].view(B, -1, C_CLIP).unsqueeze(0).repeat(K, 1, 1, 1)
        else:
            x_clip = x_clip + clip_pos_embed.view(B, -1, C_CLIP).unsqueeze(0).repeat(K, 1, 1, 1)

        x_clip_align = []
        for idx, clip_decoder in enumerate(self.clip_decoder):
            x_clip_align.append(clip_decoder(x_clip[idx]))
        x_clip_align = torch.stack(x_clip_align)

        return x_vis, x_pool_vis, x_clip_align, x_align


def interpolate_pos_embed_internvideo2(state_dict, model, orig_t_size=8):
    """Interpolate temporal position embeddings if num_frames differs."""
    key = "pos_embed"
    if key in state_dict:
        pos_embed_checkpoint = state_dict[key]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches

        orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens) // orig_t_size) ** 0.5)
        new_size = int((num_patches // model.num_frames) ** 0.5)

        if orig_t_size != model.num_frames:
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_t_size, orig_size, orig_size, embedding_size)
            pos_tokens = pos_tokens.reshape(-1, orig_t_size, orig_size * orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(model.num_frames, orig_size * orig_size),
                mode='bicubic', align_corners=False
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            state_dict[key] = new_pos_embed

    # Same for clip_pos_embed
    clip_key = "clip_pos_embed"
    if clip_key in state_dict:
        pos_embed_checkpoint = state_dict[clip_key]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.clip_pos_embed.shape[-2] - num_patches
        orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens) // orig_t_size) ** 0.5)

        if orig_t_size != model.num_frames:
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_t_size, orig_size, orig_size, embedding_size)
            pos_tokens = pos_tokens.reshape(-1, orig_t_size, orig_size * orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(model.num_frames, orig_size * orig_size),
                mode='bicubic', align_corners=False
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            state_dict[clip_key] = new_pos_embed


# ============================================================
# Video processing utilities
# ============================================================

def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        try:
            num, denom = frac_str.split('/')
        except ValueError:
            return None
        try:
            leading, num = num.split(' ')
        except ValueError:
            return float(num) / float(denom)
        if float(leading) < 0:
            sign_mult = -1
        else:
            sign_mult = 1
        return float(leading) + sign_mult * (float(num) / float(denom))


class VideoProcessor:
    """Load video frames at specified FPS using ffmpeg."""

    def __init__(self, framerate=1, size=224, centercrop=True):
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate

    def _get_video_info(self, video_path):
        probe = ffmpeg.probe(video_path)
        video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        fps = math.floor(convert_to_float(video_stream['avg_frame_rate']))
        try:
            frames_length = int(video_stream['nb_frames'])
            duration = float(video_stream['duration'])
        except Exception:
            frames_length, duration = -1, -1
        return {"duration": duration, "frames_length": frames_length,
                "fps": fps, "height": height, "width": width}

    def _get_output_dim(self, h, w):
        if isinstance(self.size, tuple) and len(self.size) == 2:
            return self.size
        elif h >= w:
            return int(h * self.size / w), self.size
        else:
            return self.size, int(w * self.size / h)

    def read_video_from_file(self, video_path):
        try:
            info = self._get_video_info(video_path)
            h, w = info["height"], info["width"]
        except Exception:
            print(f'ffprobe failed at: {video_path}')
            return None
        height, width = self._get_output_dim(h, w)
        try:
            duration = info["duration"]
            fps = self.framerate
            if duration > 0 and duration < 1 / fps + 0.1:
                fps = 2 / max(int(duration), 1)
        except Exception:
            fps = self.framerate
        cmd = (
            ffmpeg
            .input(video_path)
            .filter('fps', fps=fps)
            .filter('scale', width, height)
        )
        if self.centercrop:
            x = int((width - self.size) / 2.0)
            y = int((height - self.size) / 2.0)
            cmd = cmd.crop(x, y, self.size, self.size)
        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
        )
        if self.centercrop and isinstance(self.size, int):
            height, width = self.size, self.size
        video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
        return video  # (T, H, W, 3) uint8


# ============================================================
# Feature Extraction
# ============================================================

def build_model(checkpoint_path, num_frames=8, device='cuda'):
    """Build InternVideo2-L/14 model and load weights."""
    model = PretrainInternVideo2(
        in_chans=3, img_size=224, patch_size=14,
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        clip_embed_dim=768,
        attn_pool_num_heads=16, qkv_bias=False,
        drop_path_rate=0.0,
        init_values=1e-5,
        qk_normalization=True,
        layerscale_no_force_fp32=False,
        num_frames=num_frames, tubelet_size=1,
        sep_pos_embed=False,
        sep_image_video_pos_embed=False,
        use_checkpoint=False, checkpoint_num=0,
        clip_teacher_embed_dim=1408,
        clip_teacher_final_dim=768,
        clip_norm_type='l2',
        clip_return_layer=6,
        clip_student_return_interval=1,
    )

    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location='cpu')

    # Handle wrapped checkpoints
    if isinstance(state_dict, dict):
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'module' in state_dict:
            state_dict = state_dict['module']

    interpolate_pos_embed_internvideo2(state_dict, model, orig_t_size=8)
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"load_state_dict: {msg}")

    model = model.to(device).half()  # Use float16 to save GPU memory
    model.eval()
    return model


# ImageNet normalization
V_MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
V_STD = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)


def preprocess_frames(frames_uint8, num_frames=8):
    """
    Preprocess a batch of uint8 frames into model input tensor.

    Args:
        frames_uint8: np.array (T, H, W, 3) uint8, already center-cropped to 224x224
        num_frames: number of frames for model input

    Returns:
        tensor: (1, C, num_frames, H, W) float32 normalized
    """
    # Normalize to [0,1] then ImageNet stats
    frames = frames_uint8.astype(np.float32) / 255.0
    frames = (frames - V_MEAN) / V_STD  # (T, H, W, 3)
    # To (T, 3, H, W)
    frames = np.transpose(frames, (0, 3, 1, 2))  # (T, 3, H, W)

    T = frames.shape[0]
    if T >= num_frames:
        # Uniform sample num_frames
        indices = np.linspace(0, T - 1, num_frames, dtype=int)
        frames = frames[indices]
    else:
        # Pad by repeating last frame
        pad = np.repeat(frames[-1:], num_frames - T, axis=0)
        frames = np.concatenate([frames, pad], axis=0)

    # (1, 3, num_frames, H, W)
    tensor = torch.from_numpy(frames).unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()
    # actually already (1, T, 3, H, W) -> need (1, 3, T, H, W)
    # After permute(0, 2, 1, 3, 4): (1, 3, T, H, W) ✓
    return tensor


@torch.no_grad()
def extract_video_features(model, all_frames_uint8, num_frames=8, device='cuda', batch_size=4):
    """
    Extract per-second features using sliding window with batching.

    For each second t, we gather a window of num_frames centered on t,
    producing one 768-dim attention-pooled feature per second.

    Args:
        model: InternVideo2 model
        all_frames_uint8: (T_total, 224, 224, 3) uint8
        num_frames: temporal window size
        device: torch device
        batch_size: number of windows to process in parallel

    Returns:
        features: np.array (T_total, 768) float16
    """
    T_total = all_frames_uint8.shape[0]
    if T_total == 0:
        return None

    # Normalize all frames at once
    all_frames_f = all_frames_uint8.astype(np.float32) / 255.0
    all_frames_f = (all_frames_f - V_MEAN) / V_STD
    all_frames_f = np.transpose(all_frames_f, (0, 3, 1, 2))  # (T, 3, H, W)

    # Build all windows
    windows = []
    for t in range(T_total):
        half = num_frames // 2
        start = max(0, t - half)
        end = start + num_frames
        if end > T_total:
            end = T_total
            start = max(0, end - num_frames)

        window = all_frames_f[start:end]  # (<=num_frames, 3, H, W)
        T_w = window.shape[0]
        if T_w < num_frames:
            pad = np.repeat(window[-1:], num_frames - T_w, axis=0)
            window = np.concatenate([window, pad], axis=0)
        windows.append(window)  # (num_frames, 3, H, W)

    all_features = []

    # Process in batches
    for i in range(0, T_total, batch_size):
        batch_windows = windows[i:i + batch_size]
        # Stack: (B, num_frames, 3, H, W) -> (B, 3, num_frames, H, W)
        batch_np = np.stack(batch_windows, axis=0)
        batch_tensor = torch.from_numpy(batch_np).permute(0, 2, 1, 3, 4).contiguous()
        batch_tensor = batch_tensor.to(device).half()

        x_vis, x_pool_vis, x_clip_align, x_align = model(batch_tensor, mask=None, use_image=False)
        feats = x_pool_vis.cpu().to(torch.float16).numpy()  # (B, 768)
        all_features.append(feats)

    features = np.concatenate(all_features, axis=0)  # (T_total, 768)
    return features


# ============================================================
# Main
# ============================================================

def is_file_present(directory, vid):
    return os.path.exists(join(directory, f"{vid}.npz"))


def find_video_path(vid, input_dirs):
    for d in input_dirs:
        path = join(d, f"{vid}.mp4")
        if os.path.exists(path):
            return path
    return None


def read_pending_files(input_dirs, output_dir):
    all_vids = set()
    for input_dir in input_dirs:
        if os.path.exists(input_dir):
            vids = [os.path.splitext(f)[0] for f in os.listdir(input_dir) if f.endswith(".mp4")]
            all_vids.update(vids)
    pending = [vid for vid in sorted(all_vids) if not is_file_present(output_dir, vid)]
    return pending, len(all_vids)


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Build model
    model = build_model(args.checkpoint, num_frames=args.num_frames, device=device)

    # Video loader at 1 fps (matching existing features)
    video_loader = VideoProcessor(framerate=args.framerate, size=224, centercrop=True)

    os.makedirs(args.output_dir, exist_ok=True)

    pending_vids, total_vids = read_pending_files(args.input_dirs, args.output_dir)
    print(f"Total: {total_vids}, Pending: {len(pending_vids)}")

    # Index-based splitting for parallel extraction across GPUs
    start_idx = args.start_idx if args.start_idx >= 0 else 0
    end_idx = args.end_idx if args.end_idx >= 0 else len(pending_vids)
    end_idx = min(end_idx, len(pending_vids))
    pending_vids = pending_vids[start_idx:end_idx]
    print(f"This worker: [{start_idx}, {end_idx}) => {len(pending_vids)} videos")

    if len(pending_vids) == 0:
        print("All videos already processed.")
        return

    failed = []
    for vid in tqdm(pending_vids, desc="InternVideo2 Video Features"):
        video_path = find_video_path(vid, args.input_dirs)
        if video_path is None:
            failed.append(vid)
            continue

        try:
            frames = video_loader.read_video_from_file(video_path)
            if frames is None or len(frames) == 0:
                failed.append(vid)
                continue

            features = extract_video_features(
                model, frames, num_frames=args.num_frames, device=device,
                batch_size=args.batch_size
            )

            if features is not None:
                np.savez_compressed(
                    join(args.output_dir, f"{vid}.npz"),
                    features=features
                )

        except Exception as e:
            print(f"Failed {vid}: {e}")
            failed.append(vid)

    if failed:
        print(f"\nFailed videos ({len(failed)}):")
        failed_path = join(args.output_dir, "failed_videos.txt")
        with open(failed_path, "w") as f:
            for vid in failed:
                f.write(f"{vid}\n")
        print(f"Saved to {failed_path}")

    print(f"Done! Processed {len(pending_vids) - len(failed)}/{len(pending_vids)} videos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract InternVideo2-L/14 video features for QVHighlights")
    parser.add_argument("--checkpoint", type=str,
                        default="../InternVideo-main/InternVideo2/model/pytorch_model.bin",
                        help="Path to InternVideo2 checkpoint")
    parser.add_argument("--input_dirs", type=str, nargs="+",
                        default=[
                            "/data1/zhangshihang/data/QV highlight/train/",
                            "/data1/zhangshihang/data/QV highlight/val/",
                            "/data1/zhangshihang/data/QV highlight/test/",
                            "/data1/zhangshihang/data/QV highlight/videos/",
                        ],
                        help="Directories containing .mp4 video files")
    parser.add_argument("--output_dir", type=str,
                        default="/data1/zhangshihang/Datasets/qvhl/features/internvideo2_video_features/",
                        help="Output directory for .npz feature files")
    parser.add_argument("--framerate", type=float, default=0.5,
                        help="Frame extraction rate (0.5 = 1 frame per 2 seconds)")
    parser.add_argument("--num_frames", type=int, default=8,
                        help="Number of frames per model input (temporal window)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for feature extraction")
    parser.add_argument("--start_idx", type=int, default=-1,
                        help="Start index into sorted pending list (inclusive). -1 = from beginning")
    parser.add_argument("--end_idx", type=int, default=-1,
                        help="End index into sorted pending list (exclusive). -1 = to end")

    args = parser.parse_args()
    main(args)
