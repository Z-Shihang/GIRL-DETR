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


#### training
bsz=128
dec_layers=3
enc_layers=3
cmi_layers=1
contrastive_align_loss_coef=0.2
seed=2018

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
--dec_layers ${dec_layers} \
--enc_layers ${enc_layers} \
--cmi_layers ${cmi_layers} \
--mr_to_hd_loss \
--hard_pos_neg_loss \
--contrastive_align_loss \
--contrastive_align_loss_coef ${contrastive_align_loss_coef} \
--results_root ${results_root} \
--exp_id exp-cmi_${cmi_layers}-en_${enc_layers}-dec_${dec_layers}-tcl-hl-scsl-cal_${contrastive_align_loss_coef}-${v_feat_types}-seed_${seed} \
--device 0 \
--hidden_dim 256 \
--seed ${seed} \
${@:1}

