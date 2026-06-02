#!/bin/bash
# SCST v7b: Curriculum Learning — Three-Phase Schedule
# Strategy: decompose SCST RL loss into gentler stages
#   Phase 1 (epoch 0~14):  top1_refine only, no RL → stabilize span boundaries
#   Phase 2 (epoch 15~39): + reward-weighted span regression (gentle, weighted L1)
#   Phase 3 (epoch 40~79): + score REINFORCE (full RL, policy gradient)
# Each phase ramps its component linearly to avoid sudden jumps.
# Base checkpoint: v5b (top-1 refinement, best supervised)
# Default GPU: 7 (set DEVICE env var to override)

dset_name=charadesSTA
ctx_mode=video_tef
v_feat_types=internvideo2_clip_blip
t_feat_types=clip_blip
results_root=results/charadesSTA/internvideo2_scst_v7b_3phase
exp_id=exp-scst-ft-v7b-3phase-inter

######## data paths
train_path=data/charades_sta/charades_sta_train_tvr_format.jsonl
eval_path=data/charades_sta/charades_sta_test_tvr_format.jsonl
eval_split_name=val

######## setup video+text features
# Dataset paths (can be overridden via environment variables)
data_root=${DATA_ROOT:-data/charades_sta}
feat_root=${FEAT_ROOT:-../Datasets/charades}

# video features
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

# text features
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

######## Gate mode must match resume checkpoint.
# Options: none | task_decouple | cmi | hd_to_mr | decoder_sent | span_gaussian
gate_mode=${GATE_MODE:-decoder_sent}
gate_args=()

case "${gate_mode}" in
  none)
default_ckpt=../GIRL_DETR/results/charadesSTA/internvideo2/charadesSTA-video_tef-exp-iv2_replace_sf_ema-2026_04_20_22_09_18/model_best.ckpt
    ;;
  task_decouple)
    gate_args+=(--task_decouple_gate)
    default_ckpt=../GIRL_DETR/results/charadesSTA/internvideo2/charadesSTA-video_tef-exp-iv2_replace_sf_ema-2026_04_20_22_41_26/model_best.ckpt
    ;;
  cmi)
    gate_args+=(--cmi_gate)
    default_ckpt=../GIRL_DETR/results/charadesSTA/internvideo2/charadesSTA-video_tef-exp-iv2_replace_sf_ema-2026_04_20_22_45_45/model_best.ckpt
    ;;
  hd_to_mr)
    gate_args+=(--hd_to_mr_gate)
    default_ckpt=../GIRL_DETR/results/charadesSTA/internvideo2/charadesSTA-video_tef-exp-iv2_replace_sf_ema-2026_04_20_22_48_51/model_best.ckpt
    ;;
  decoder_sent)
    gate_args+=(--decoder_sent_gate)
    default_ckpt=../GIRL_DETR/results/charadesSTA/internvideo2/charadesSTA-video_tef-exp-iv2_replace_sf_ema-2026_04_20_22_52_51/model_best.ckpt
    ;;
  span_gaussian)
    gate_args+=(--span_gaussian_mask)
    default_ckpt=../GIRL_DETR/results/charadesSTA/internvideo2/charadesSTA-video_tef-exp-iv2_replace_sf_ema-2026_04_20_22_53_14/model_best.ckpt
    ;;
  *)
    echo "Unknown GATE_MODE=${gate_mode}. Use: none|task_decouple|cmi|hd_to_mr|decoder_sent|span_gaussian"
    exit 1
    ;;
esac

resume_ckpt=${CKPT:-${default_ckpt}}

#### fine-tuning hyperparams
bsz=128
eval_bsz=16
lr=5e-05
lr_drop=200
dec_layers=3
enc_layers=3
cmi_layers=1
max_v_l=75
rl_coef=${RL_COEF:-0.03}
top1_coef=1.0
phase1_epochs=${PHASE1:-8}
phase2_epochs=${PHASE2:-12}
n_epoch=80
main_metric=${MAIN_METRIC:-MR-full-mIoU}
rl_phase3_decay=${RL_PHASE3_DECAY:-cosine}
device=${DEVICE:-7}

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
--rl_phase3_decay ${rl_phase3_decay} \
--scst \
--rl_only \
--top1_refine \
--freeze_backbone \
--main_metric ${main_metric} \
--ema_decay 0.999 \
--seed 3800 \
${gate_args[@]} \
--hidden_dim 256 \
--resume ${resume_ckpt} \
${@:1}
