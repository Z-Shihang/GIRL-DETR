#!/bin/bash

set -euo pipefail

dset_name=tacos
ctx_mode=video_tef
v_feat_types=${V_FEAT_TYPES:-internvideo2_clip_blip}
t_feat_types=${T_FEAT_TYPES:-clip_blip}
results_root=${RESULTS_ROOT:-results/tacos/internvideo2_scst_v15c_frozen_lowlr}
exp_id=${EXP_ID:-exp-scst-ft-v15c-frozen-lowlr}

# Dataset paths (can be overridden via environment variables)
data_root=${DATA_ROOT:-data/tacos}
train_path=${TRAIN_PATH:-${data_root}/train.jsonl}
eval_path=${EVAL_PATH:-${data_root}/val.jsonl}
eval_split_name=val

feat_root=${FEAT_ROOT:-${data_root}}

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

resume_ckpt=${CKPT:-"../GIRL_DETR/results/tacos/internvideo2/tacos-video_tef-exp-iv2-s4-2026_04_24_10_28_55/model_best.ckpt"}


bsz=${BSZ:-32}
eval_bsz=${EVAL_BSZ:-8}
lr=${LR:-2e-5}
lr_drop=${LR_DROP:-200}
dec_layers=${DEC_LAYERS:-3}
enc_layers=${ENC_LAYERS:-3}
cmi_layers=${CMI_LAYERS:-1}
max_v_l=${MAX_V_L:-1000}
clip_length=${CLIP_LENGTH:-1}
rl_coef=${RL_COEF:-0.02}
top1_coef=${TOP1_COEF:-1.0}
phase1_epochs=${PHASE1:-10}
phase2_epochs=${PHASE2:-10}
n_epoch=${N_EPOCH:-60}
contrastive_align_loss_coef=${CONTRASTIVE_ALIGN_LOSS_COEF:-0.3}
main_metric=${MAIN_METRIC:-MR-full-mIoU}
device=${DEVICE:-0}
seed=${SEED:-3800}
python_bin=${PYTHON_BIN:-python}

CUDA_VISIBLE_DEVICES=${device} PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}." \
  ${python_bin} girl/train.py \
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
  --clip_length ${clip_length} \
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
  --contrastive_align_loss_coef ${contrastive_align_loss_coef} \
  --rl_coef ${rl_coef} \
  --top1_coef ${top1_coef} \
  --progressive_rl three_phase \
  --phase1_epochs ${phase1_epochs} \
  --phase2_epochs ${phase2_epochs} \
  --scst \
  --rl_only \
  --top1_refine \
  --freeze_backbone \
  --decoder_sent_gate \
  --main_metric ${main_metric} \
  --ema_decay 0.999 \
  --hidden_dim 256 \
  --seed ${seed} \
  --resume ${resume_ckpt} \
  ${@:1}