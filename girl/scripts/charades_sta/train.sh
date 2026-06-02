dset_name=charadesSTA
ctx_mode=video_tef
v_feat_types=internvideo2_clip_blip
t_feat_types=clip_blip
results_root=results/charadesSTA/internvideo2
exp_id=exp-iv2_replace_sf_ema

######## data paths
train_path=data/charades_sta/charades_sta_train_tvr_format.jsonl
eval_path=data/charades_sta/charades_sta_test_tvr_format.jsonl
eval_split_name=val

######## setup video+text features
# Dataset paths (can be overridden via environment variables)
data_root=${DATA_ROOT:-data/charades_sta}
feat_root=${FEAT_ROOT:-../Datasets/charades}

# video features: InternVideo2(768) + CLIP(512) + BLIP(768) = 2048
v_feat_dim=0
v_feat_dirs=()
v_feat_dirs+=(${feat_root}/internvideo2_video_features)
(( v_feat_dim += 768 ))
v_feat_dirs+=(${feat_root}/clip_video_features)
(( v_feat_dim += 512 ))
v_feat_dirs+=(${feat_root}/blip_video_features)
(( v_feat_dim += 768 ))

# text features
t_feat_dim=0
t_feat_dirs=()
t_feat_dirs+=(${feat_root}/clip_text_features)
(( t_feat_dim += 512 ))
t_feat_dirs+=(${feat_root}/blip_text_features)
(( t_feat_dim += 768 ))



#### training
bsz=256
eval_bsz=32
lr=2e-04
lr_drop=100
dec_layers=3
enc_layers=3
cmi_layers=1
max_v_l=75
contrastive_align_loss_coef=0.3
hard_pos_neg_loss_coef=10
main_metric="MR-full-R1@0.3"


PYTHONPATH=$PYTHONPATH:. python girl/train.py \
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
--n_epoch 100 \
--eval_bsz ${eval_bsz} \
--dec_layers ${dec_layers} \
--enc_layers ${enc_layers} \
--cmi_layers ${cmi_layers} \
--mr_to_hd_loss \
--hard_pos_neg_loss \
--hard_pos_neg_loss_coef ${hard_pos_neg_loss_coef} \
--contrastive_align_loss \
--contrastive_align_loss_coef ${contrastive_align_loss_coef} \
--main_metric ${main_metric} \
--ema_decay 0.999 \
--seed 3800 \
${@:1}
