#!/bin/bash
# SCST v15c: Three-Phase — frozen backbone, lower lr + lower rl_coef
#
# Same frozen-backbone setup as v7b (only span_embed/class_embed/iou_embed trainable)
# but Phase 2 uses lower rl_coef=0.02 and lower lr=2e-5, so Phase 2 doesn't fully saturate
# → Phase 3 GRPO starts from ~37.4 (not the 37.69 peak) and has room to improve
#
# Phase 1 (epoch 0~9):   top1_refine only (10 epochs, SL warmup)
# Phase 2 (epoch 10~19): RWR, rl_coef=0.02 (10 epochs, partial convergence)
# Phase 3 (epoch 20~59): GRPO full RL (40 epochs, starts from non-saturated Phase 2)
#
# Key difference from v7b: rl_coef 0.03→0.02, lr 5e-5→2e-5 → Phase 2 not saturated
# No decoder unfreeze (safe, consistent with prior successful experiments)

dset_name=charadesSTA
ctx_mode=video_tef
v_feat_types=internvideo2_clip_blip
t_feat_types=clip_blip
results_root=results/charadesSTA/internvideo2_scst_v15c_frozen_lowlr
exp_id=exp-scst-ft-v15c-frozen-lowlr

######## data paths
train_path=data/charades_sta/charades_sta_train_tvr_format.jsonl
eval_path=data/charades_sta/charades_sta_test_tvr_format.jsonl
eval_split_name=val

######## setup video+text features
# Dataset paths (can be overridden via environment variables)
data_root=${DATA_ROOT:-data/charades_sta}
feat_root=${FEAT_ROOT:-../Datasets/charades}

v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"internvideo2"* ]]; then
  v_feat_dirs+=(${feat_root}/internvideo2_video_features)
  (( v_feat_dim += 768 ))
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip_video_features)
  (( v_feat_dim += 512 ))
fi
if [[ ${v_feat_types} == *"blip"* ]]; then
  v_feat_dirs+=(${feat_root}/blip_video_features)
  (( v_feat_dim += 768 ))
fi

t_feat_dim=0
t_feat_dirs=()
if [[ ${t_feat_types} == *"clip"* ]]; then
  t_feat_dirs+=(${feat_root}/clip_text_features)
  (( t_feat_dim += 512 ))
fi
if [[ ${t_feat_types} == *"blip"* ]]; then
  t_feat_dirs+=(${feat_root}/blip_text_features)
  (( t_feat_dim += 768 ))
fi

resume_ckpt=${CKPT:-../GIRL_DETR/results/charadesSTA/internvideo2/charadesSTA-video_tef-exp-iv2_replace_sf_ema-2026_04_20_22_09_18/model_best.ckpt}

bsz=128
eval_bsz=16
lr=${LR:-5e-05}             # lower than v7b (5e-5) → slower convergence → Phase 2 not saturated
lr_drop=200
dec_layers=3
enc_layers=3
cmi_layers=1
max_v_l=75
rl_coef=${RL_COEF:-0.03}    # lower than v7b (0.03) → Phase 2 RWR signal is gentler
top1_coef=1.0
phase1_epochs=${PHASE1:-20}
phase2_epochs=${PHASE2:-20}
n_epoch=80
main_metric=${MAIN_METRIC:-MR-full-mIoU}
device=${DEVICE:-1}

CUDA_VISIBLE_DEVICES=${device} PYTHONPATH=$PYTHONPATH:. \
  python girl/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dirs ${t_feat_dirs[@]} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id ${exp_id} \
--max_v_l ${max_v_l} \
--clip_length 1 \
--lr ${lr} \
--lr_drop ${lr_drop} \
--n_epoch ${n_epoch} \
--eval_bsz ${eval_bsz} \
--dec_layers ${dec_layers} \
--enc_layers ${enc_layers} \
--cmi_layers ${cmi_layers} \
--mr_to_hd_loss \
--hard_pos_neg_loss \
--hard_pos_neg_loss_coef 10 \
--contrastive_align_loss \
--contrastive_align_loss_coef 0.3 \
--rl_coef ${rl_coef} \
--top1_coef ${top1_coef} \
--progressive_rl three_phase \
--phase1_epochs ${phase1_epochs} \
--phase2_epochs ${phase2_epochs} \
--scst \
--rl_only \
--top1_refine \
--freeze_backbone \
--main_metric ${main_metric} \
--ema_decay 0.999 \
--seed 3800 \
--hidden_dim 256 \
--resume ${resume_ckpt} \
${@:1}
