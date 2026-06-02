#!/bin/bash
# SCST v15c for QVHighlights: Three-Phase — frozen backbone, lower lr + lower rl_coef
#
# QVHighlights version of charades_sta/finetune_scst_v15c_frozen_lowlr.sh
#
# Strategy: same as Charades v15c
#   Phase 1 (epoch 0~9):   top1_refine only (10 epochs, SL warmup)
#   Phase 2 (epoch 10~19): RWR, rl_coef=0.02 (10 epochs, partial convergence)
#   Phase 3 (epoch 20~59): GRPO full RL (40 epochs, starts from non-saturated Phase 2)
#
# Key differences from v7b: rl_coef 0.03→0.02, lr 5e-5→2e-5 → Phase 2 not saturated
# No decoder unfreeze (safe, consistent with prior successful experiments)
# Resume from decoder_sent_gate best QVHighlights checkpoint
dset_name=hl
ctx_mode=video_tef
#v_feat_types=clip
#v_feat_types=slowfast_clip
v_feat_types=internvideo2_clip_blip
#t_feat_types=clip
t_feat_types=clip_blip
results_root=results/qvhighlights/Final
exp_id=exp

######## data paths
train_path=data/highlight_train_release.jsonl
#train_path=data/highlight_train_release_paraphrased.jsonl
#train_path=data/highlight_train_release_paraphrased_openai.jsonl
eval_path=data/highlight_val_release.jsonl
eval_split_name=val

######## setup video+text features
feat_root=../Datasets/qvhl/features


v_feat_dim=0
v_feat_dirs=()
v_feat_dirs+=(${feat_root}/internvideo2_video_features)
(( v_feat_dim += 768 ))
v_feat_dirs+=(${feat_root}/clip_features)
(( v_feat_dim += 512 ))
v_feat_dirs+=(${feat_root}/blip_video_features)
(( v_feat_dim += 768 ))

# text features
t_feat_dim=0
t_feat_dirs=()
t_feat_dirs+=(${feat_root}/clip_text_features)
(( t_feat_dim += 512 ))
t_feat_dirs+=(${feat_root}/blip_aug_text_features_openai)
(( t_feat_dim += 768 ))

# Resume from decoder_sent_gate QVHighlights best checkpoint (selected by MR-full-mAP)
# 10_17_08: mAP=49.57, R1@0.5=71.94, R1@0.7=56.71 ← best mAP (main_metric)
# 10_46_56: mAP=49.38, R1@0.5=72.13, R1@0.7=56.77
default_ckpt="../GIRL_DETR/results/qvhighlights/Final/hl-video_tef-exp-cmi_1-en_3-dec_3-tcl-hl-scsl-cal_0.2-internvideo2_clip_blip-seed_2018-sentgate_full-2026_04_23_10_46_56/model_best.ckpt"
#default_ckpt="results/qvhighlights/Final/hl-video_tef-exp-cmi_1-en_3-dec_3-tcl-hl-scsl-cal_0.2-slowfast_clip_blip-seed_2018-sentgate_full-2026_05_02_20_16_45/model_best.ckpt"
resume_ckpt=${CKPT:-${default_ckpt}}

bsz=128
eval_bsz=16
lr=${LR:-2e-04}             # lower than v7b (5e-5) → slower convergence → Phase 2 not saturated
lr_drop=200
dec_layers=3
enc_layers=3
cmi_layers=1
max_v_l=75
rl_coef=${RL_COEF:-0.05}    # lower than v7b (0.03) → Phase 2 RWR signal is gentler
top1_coef=1.0
phase1_epochs=${PHASE1:-20}
phase2_epochs=${PHASE2:-20}
n_epoch=60
contrastive_align_loss_coef=0.2
main_metric=${MAIN_METRIC:-MR-full-mAP}
device=${DEVICE:-0}
seed=2018

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
--freeze_backbone \
--rl_only \
--top1_refine \
--main_metric ${main_metric} \
--ema_decay 0.999 \
--seed ${seed} \
--decoder_sent_gate \
--hidden_dim 256 \
--resume ${resume_ckpt} \
${@:1}
