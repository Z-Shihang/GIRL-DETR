import os
import time
import torch
import argparse

from utils.basic_utils import mkdirp, load_json, save_json, make_zipfile, dict_to_markdown
import shutil

def backward_compat_param_names(opt_dict):
    """Convert old parameter names to new names for backward compatibility.
    
    This function handles renamed parameters from code refactoring.
    Old parameter names from checkpoints are automatically converted to new names.
    """
    param_renames = {
        'bicmf_layers': 'cmi_layers',
        'bicmf_gate': 'cmi_gate',
        'unfreeze_bicmf': 'unfreeze_cmi',
        'rl_curriculum': 'progressive_rl',
    }
    
    renamed_any = False
    for old_name, new_name in param_renames.items():
        if old_name in opt_dict:
            opt_dict[new_name] = opt_dict.pop(old_name)
            renamed_any = True
    
    return renamed_any

class BaseOptions(object):
    saved_option_filename = "opt.json"
    ckpt_filename = "model.ckpt"
    tensorboard_log_dir = "tensorboard_log"
    train_log_filename = "train.log.txt"
    eval_log_filename = "eval.log.txt"

    def __init__(self):
        self.parser = None
        self.initialized = False
        self.opt = None

    def initialize(self):
        self.initialized = True
        parser = argparse.ArgumentParser()
        parser.add_argument("--dset_name", default="hl", type=str, choices=['hl', 'tvsum', 'charadesSTA',
                                                                            'tacos', 'youtube_uni'])
        parser.add_argument("--dset_domain", type=str, default='BK',
                            help="Domain to train for tvsum/youtube-hl dataset.")
        
        parser.add_argument("--eval_split_name", type=str, default="val",
                            help="should match keys in video_duration_idx_path, must set for VCMR")
        parser.add_argument("--main_metric", type=str, default="MR-full-mAP",
                            help="Main metric that wil be used to determine the best model")
        parser.add_argument("--best_metric_strategy", type=str, default="weighted_stop_score",
                            choices=["weighted_stop_score", "main_metric"],
                            help="How to select the best checkpoint during training. "
                                 "weighted_stop_score keeps the historical mixed score, "
                                 "main_metric uses --main_metric directly.")
        parser.add_argument("--debug", action="store_true",
                            help="debug (fast) mode, break all loops, do not load all data into memory.")
        parser.add_argument("--data_ratio", type=float, default=1.0,
                            help="how many training and eval data to use. 1.0: use all, 0.1: use 10%."
                                 "Use small portion for debug purposes. Note this is different from --debug, "
                                 "which works by breaking the loops, typically they are not used together.")
        parser.add_argument("--results_root", type=str, default="results")
        parser.add_argument("--exp_id", type=str, default="debug_exp", help="id of this run, required at training")
        parser.add_argument("--seed", type=int, default=3800, help="random seed")
        parser.add_argument("--device", type=int, default=0, help="0 cuda, -1 cpu")
        parser.add_argument("--num_workers", type=int, default=4,
                            help="num subprocesses used to load the data, 0: use main process")
        parser.add_argument("--no_pin_memory", action="store_true",
                            help="Don't use pin_memory=True for dataloader. "
                                 "ref: https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4")

        # training config
        parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
        parser.add_argument("--lr_heads", type=float, default=None,
                            help="Learning rate for prediction heads during frozen-backbone fine-tuning. "
                                 "If unset, falls back to --lr.")
        parser.add_argument("--lr_unfrozen", type=float, default=None,
                            help="Learning rate for newly unfrozen non-head modules during frozen-backbone "
                                 "fine-tuning. If unset, falls back to --lr.")
        parser.add_argument("--lr_drop", type=int, default=400, help="drop learning rate to 1/10 every lr_drop epochs")
        parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
        parser.add_argument("--warmup_epochs", type=int, default=0, help="number of warmup epochs (0 to disable)")
        parser.add_argument("--lr_scheduler", type=str, default="step", choices=["step", "cosine"],
                            help="lr scheduler type: step (StepLR) or cosine (CosineAnnealingLR)")
        parser.add_argument("--ema_decay", type=float, default=0.0,
                            help="EMA decay rate. 0 to disable EMA, typical value 0.999 or 0.9999")
        parser.add_argument("--n_epoch", type=int, default=200, help="number of epochs to run")
        parser.add_argument("--max_es_cnt", type=int, default=200,
                            help="number of epochs to early stop, use -1 to disable early stop")
        parser.add_argument("--bsz", type=int, default=32, help="mini-batch size")
        parser.add_argument("--eval_bsz", type=int, default=100,
                            help="mini-batch size at inference, for query")
        parser.add_argument("--grad_clip", type=float, default=0.1, help="perform gradient clip, -1: disable")
        parser.add_argument("--eval_untrained", action="store_true", help="Evaluate on un-trained model")
        parser.add_argument("--resume", type=str, default=None,
                            help="checkpoint path to resume or evaluate, without --resume_all this only load weights")
        parser.add_argument("--resume_all", action="store_true",
                            help="if --resume_all, load optimizer/scheduler/epoch as well")
        # SCST RL fine-tuning
        parser.add_argument("--scst", action="store_true",
                            help="Enable SCST (Self-Critical Sequence Training) RL fine-tuning")
        parser.add_argument("--rl_coef", type=float, default=0.1,
                            help="Weight for SCST RL loss (combined with supervised loss)")
        parser.add_argument("--rl_only", action="store_true",
                            help="Use ONLY RL loss, no supervised loss (pure SCST fine-tuning)")
        parser.add_argument("--freeze_backbone", action="store_true",
                            help="Freeze the backbone; only prediction heads and explicitly unfreezed modules remain trainable")
        parser.add_argument("--top1_refine", action="store_true",
                            help="Enable top-1 query targeted refinement loss")
        parser.add_argument("--top1_coef", type=float, default=1.0,
                            help="Weight for top-1 refinement loss")
        parser.add_argument("--use_iou_head", action="store_true",
                            help="Enable IoU prediction head for scoring (must be trained, not randomly init)")
        parser.add_argument("--iou_rerank_alpha", type=float, default=0.0,
                            help="Blend fg_score and pred_iou at inference: score = (1-alpha)*fg_score + alpha*pred_iou. "
                                 "0=pure fg_score, 1=pure pred_iou. Recommended: 0.3~0.7. "
                                 "Requires model trained with iou loss (pred_iou available).")
        # Curriculum learning for RL
        parser.add_argument("--rl_warmup_epochs", type=int, default=0,
                            help="Number of epochs to linearly warmup rl_coef from 0 to target. 0=no warmup.")
        parser.add_argument("--progressive_rl", type=str, default="none",
                            choices=["none", "linear", "three_phase"],
                            help="Progressive RL schedule: none=fixed rl_coef, linear=warmup rl_coef, "
                                 "three_phase=span_only→full_rl→full")
        parser.add_argument("--phase1_epochs", type=int, default=15,
                            help="[three_phase] Epochs for phase1 (top1_refine only, no RL)")
        parser.add_argument("--phase2_epochs", type=int, default=25,
                            help="[three_phase] Epochs for phase2 (+ reward-weighted span regression)")
        parser.add_argument("--threshold_aware_reward", action="store_true",
                            help="Use threshold-aware weighting in span regression: upweight queries near IoU=0.7")
        parser.add_argument("--span_weight_mode", type=str, default="improvement",
                            choices=["improvement", "balanced", "iou"],
                            help="Span regression weighting: improvement=(1-IoU), balanced=IoU*(1-IoU), iou=IoU")
        parser.add_argument("--unfreeze_decoder_last", action="store_true",
                            help="Unfreeze the last decoder layer in addition to the prediction heads")
        parser.add_argument("--unfreeze_decoder_all", action="store_true",
                            help="Unfreeze ALL decoder layers in addition to the prediction heads")
        parser.add_argument("--unfreeze_query_embed", action="store_true",
                            help="Unfreeze query embeddings in addition to the prediction heads")
        parser.add_argument("--unfreeze_decoder_geometry", action="store_true",
                            help="Unfreeze decoder geometry modules (bbox_embed/ref_point_head/query_scale/ref_anchor_head)")
        parser.add_argument("--unfreeze_decoder_gate", action="store_true",
                            help="Unfreeze decoder sentence-gate modules when --decoder_sent_gate is enabled")
        parser.add_argument("--unfreeze_global_tokens", action="store_true",
                            help="Unfreeze the learnable global token and global positional token")
        parser.add_argument("--unfreeze_input_proj", action="store_true",
                            help="Unfreeze input projection layers (input_vid_proj/input_txt_proj)")
        parser.add_argument("--unfreeze_feature_refinement", action="store_true",
                            help="Unfreeze the FeatureRefinement block before the transformer")
        parser.add_argument("--unfreeze_saliency_proj", action="store_true",
                            help="Unfreeze saliency heads (saliency_proj1/2 and saliency_proj_mr if present)")
        parser.add_argument("--unfreeze_encoder_last", action="store_true",
                            help="Unfreeze the last transformer encoder layer")
        parser.add_argument("--unfreeze_cmi", action="store_true",
                            help="Unfreeze the CMI text-to-video encoder (higher risk, changes cross-modal interaction)")
        parser.add_argument("--rl_n_samples", type=int, default=3,
                            help="Number of samples for REINFORCE in SCST")
        parser.add_argument("--scst_giou", action="store_true",
                            help="Add GIoU loss to SCST span regression (in addition to L1)")
        parser.add_argument("--rl_phase3_decay", type=str, default="none",
                            choices=["none", "cosine"],
                            help="Decay schedule for rl_coef in Phase3: cosine decays from rl_coef to rl_coef/5")
        parser.add_argument("--scst_boundary_bonus", action="store_true",
                            help="Add boundary precision bonus to SCST reward (Gaussian penalty for start/end error)")
        parser.add_argument("--scst_threshold_steps", action="store_true",
                            help="Add stepping stone bonuses at IoU thresholds (0.3, 0.5, 0.7) to SCST reward")
        # Phase3 design fixes
        parser.add_argument("--phase3_anchor_weight", type=float, default=0.0,
                            help="[Option-A] In Phase3, add supervised loss * this weight as anchor to prevent RL drift. "
                                 "0.0=disabled (original), recommended 0.1-0.3")
        parser.add_argument("--scst_iou_greedy", action="store_true",
                            help="[Option-B] Use IoU-best query as REINFORCE baseline instead of score-best query. "
                                 "Makes advantage = (sampled_iou - best_iou), more meaningful for MR task.")
        parser.add_argument("--scst_baseline_mode", type=str, default="score_greedy",
                            choices=["score_greedy", "iou_greedy", "loo_mean", "grpo"],
                            help="[Option-B/C/D] REINFORCE baseline: score_greedy=score-argmax (default), "
                                 "iou_greedy=IoU-argmax (Option-B), loo_mean=mean of all query rewards (Option-C), "
                                 "grpo=vectorized GRPO advantage over all queries (Option-D, recommended)")
        parser.add_argument("--scst_reward_mode", type=str, default="iou",
                            choices=["iou", "step", "iou_step"],
                            help="Reward shaping: iou=raw IoU (default), "
                                 "step=0.5*(IoU>0.5)+0.5*(IoU>0.7) binary thresholds, "
                                 "iou_step=raw IoU + threshold bonuses")
        parser.add_argument("--scst_clip_adv", type=float, default=0.0,
                            help="Clip GRPO advantages to [-clip_adv, clip_adv]. 0=disabled. Recommended: 2.0 with grpo mode")
        parser.add_argument("--scst_ic_reward_weight", type=float, default=0.0,
                            help="Weight for Intersection Compactness reward (VTG-Reasoner). "
                                 "R_IC = |GT∩pred|/|pred|, only active when IoU>=0.5. "
                                 "Penalizes over-wide predictions. Recommended: 0.3~0.5. 0=disabled.")
        parser.add_argument("--scst_cendist_weight", type=float, default=0.0,
                            help="Weight for Center Distance reward (LongVTG-R1). "
                                 "R_CenDist = 1 - |c_pred - c_gt| / Duration. "
                                 "Provides dense gradient even when IoU=0 (blended: base*(1-w) + cendist*w). "
                                 "Recommended: 0.2. 0=disabled.")
        parser.add_argument("--scst_top_k", type=int, default=3,
                            help="Number of top foreground queries for span regression in SCST. "
                                 "Default 3. Increase (e.g. 5~7) to give more queries gradient signal, "
                                 "especially useful when many predictions cluster near IoU thresholds.")
        parser.add_argument("--phase3_span_only", action="store_true",
                            help="[three_phase] In Phase 3, use ONLY span regression (no score REINFORCE). "
                                 "Equivalent to an enhanced Phase 2 with potentially different hyperparameters. "
                                 "Useful when REINFORCE is unstable or not needed after Phase 2 span training.")
        # Score-IoU ranking loss
        parser.add_argument("--score_iou_rank_coef", type=float, default=0.0,
                            help="Coefficient for score-IoU pairwise ranking loss. 0=disabled. "
                                 "Trains foreground scores to correlate with IoU so top-1 by score ≈ top-1 by IoU. "
                                 "Recommended: 0.5~2.0")
        parser.add_argument("--score_iou_rank_margin", type=float, default=0.1,
                            help="Margin for pairwise ranking loss (default 0.1)")
        parser.add_argument("--score_iou_rank_iou_threshold", type=float, default=0.1,
                            help="Minimum IoU difference to trigger ranking loss between a query pair (default 0.1)")
        parser.add_argument("--start_epoch", type=int, default=None,
                            help="if None, will be set automatically when using --resume_all")

        # Data config
        parser.add_argument("--max_q_l", type=int, default=32)
        parser.add_argument("--max_v_l", type=int, default=75)
        parser.add_argument("--clip_length", type=int, default=2)
        parser.add_argument("--max_windows", type=int, default=5)

        parser.add_argument("--train_path", type=str, default="data/highlight_train_release.jsonl")
        parser.add_argument("--eval_path", type=str, default="data/highlight_val_release.jsonl",
                            help="Evaluating during training, for Dev set. If None, will only do training, ")
        parser.add_argument("--no_norm_vfeat", action="store_true", help="Do not do normalize video feat")
        parser.add_argument("--no_norm_tfeat", action="store_true", help="Do not do normalize text feat")
        parser.add_argument("--v_feat_dirs", type=str, nargs="+",
                            default=[
                                "../Datasets/qvhl/features/slowfast_features",
                                "../Datasets/qvhl/features/clip_features"
                            ],
                            help="video feature dirs. If more than one, will concat their features. "
                                 "Note that sub ctx features are also accepted here.")
        parser.add_argument("--t_feat_dirs", type=str, nargs="+",
                            default=[
                                "../Datasets/qvhl/features/clip_aug_text_features_openai/",
                                "../Datasets/qvhl/features/blip_aug_text_features_openai/"
                            ],
                            # default=[
                            #             "../Datasets/processed/charades/clip_text_features/",
                            #             # "../Datasets/processed/charades/blip_text_features/"
                            #         ],
                            help="text/query feature dir")
        # parser.add_argument("--t_feat_dir", type=str,
        #                     default="../Datasets/qvhl/features/clip_aug_text_features_openai/",
        #                     help="text/query feature dir")
        parser.add_argument("--a_feat_dir", type=str, help="audio feature dir")
        parser.add_argument("--v_feat_dim", type=int, default=2816, help="video feature dim")
        parser.add_argument("--t_feat_dim", type=int, default=1280, help="text/query feature dim")
        parser.add_argument("--a_feat_dim", type=int, default=0, help="audio feature dim")
        parser.add_argument("--ctx_mode", type=str, default="video_tef")

        # Model config
        parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                            help="Type of positional embedding to use on top of the image features")
        # * Transformer
        parser.add_argument('--cmi_layers', default=2, type=int,
                            help="Number of cmi layers in the transformer")
        parser.add_argument('--enc_layers', default=2, type=int,
                            help="Number of encoding layers in the transformer")
        parser.add_argument('--dec_layers', default=2, type=int,
                            help="Number of decoding layers in the transformer")
        parser.add_argument('--dim_feedforward', default=1024, type=int,
                            help="Intermediate size of the feedforward layers in the transformer blocks")
        parser.add_argument('--hidden_dim', default=256, type=int,
                            help="Size of the embeddings (dimension of the transformer)")
        parser.add_argument('--input_dropout', default=0.5, type=float,
                            help="Dropout applied in input")
        parser.add_argument('--dropout', default=0.1, type=float,
                            help="Dropout applied in the transformer")
        parser.add_argument("--txt_drop_ratio", default=0, type=float,
                            help="drop txt_drop_ratio tokens from text input. 0.1=10%")
        parser.add_argument("--use_txt_pos", action="store_true", help="use position_embedding for text as well.")
        parser.add_argument('--nheads', default=8, type=int,
                            help="Number of attention heads inside the transformer's attentions")
        parser.add_argument('--num_queries', default=10, type=int,
                            help="Number of query slots")
        parser.add_argument('--pre_norm', action='store_true')
        # other model configs
        parser.add_argument("--n_input_proj", type=int, default=2, help="#layers to encoder input")
        parser.add_argument("--contrastive_hdim", type=int, default=64, help="dim for contrastive embeddings")
        parser.add_argument("--temperature", type=float, default=0.07, help="temperature nce contrastive_align_loss")
        # Loss
        parser.add_argument("--lw_saliency", type=float, default=1.,
                            help="weight for saliency loss, set to 0 will ignore")
        parser.add_argument("--saliency_margin", type=float, default=0.2)
        parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                            help="Disables auxiliary decoding losses (loss at each layer)")
        parser.add_argument("--span_loss_type", default="l1", type=str, choices=['l1', 'ce'],
                            help="l1: (center-x, width) regression. ce: (st_idx, ed_idx) classification.")
        parser.add_argument("--contrastive_align_loss", action="store_true",
                            help="Disable contrastive_align_loss between matched query spans and the text.")
        parser.add_argument("--hard_pos_neg_loss", action="store_true", #default=True,
                            help="Enable hard positive and negative saliency loss.")
        parser.add_argument("--mr_to_hd_loss", action="store_true", #default=True,
                            help="Enable mr_to_hd_loss.")
        parser.add_argument("--decoder_sent_gate", action="store_true",
                            help="Enable EaTR-style sentence-level gated fusion in decoder first layer.")
        parser.add_argument("--no_feature_refinement", action="store_true",
                            help="Ablation: disable the cross-modal FeatureRefinement module before the transformer encoder.")

        # * Matcher
        parser.add_argument('--set_cost_span', default=10, type=float,
                            help="L1 span coefficient in the matching cost")
        parser.add_argument('--set_cost_giou', default=1, type=float,
                            help="giou span coefficient in the matching cost")
        parser.add_argument('--set_cost_class', default=4, type=float,
                            help="Class coefficient in the matching cost")

        # * Loss coefficients
        parser.add_argument('--span_loss_coef', default=10, type=float)
        parser.add_argument('--giou_loss_coef', default=1, type=float)
        parser.add_argument('--label_loss_coef', default=4, type=float)
        parser.add_argument('--iou_loss_coef', default=1.0, type=float,
                            help="IoU-aware prediction loss coefficient")
        parser.add_argument('--eos_coef', default=0.1, type=float,
                            help="Relative classification weight of the no-object class")
        parser.add_argument("--contrastive_align_loss_coef", default=0.2, type=float)
        parser.add_argument("--hard_pos_neg_loss_coef", default=10.0, type=float)
        parser.add_argument("--cos_sim_loss_coef", default=1.0, type=float)
        parser.add_argument("--mr_to_hd_loss_coef", default=1.0, type=float)

        parser.add_argument("--no_sort_results", action="store_true",
                            help="do not sort results, use this for moment query visualization")
        parser.add_argument("--max_before_nms", type=int, default=10)
        parser.add_argument("--max_after_nms", type=int, default=10)
        parser.add_argument("--conf_thd", type=float, default=0.0, help="only keep windows with conf >= conf_thd")
        parser.add_argument("--nms_thd", type=float, default=-1,
                            help="additionally use non-maximum suppression "
                                 "(or non-minimum suppression for distance)"
                                 "to post-processing the predictions. "
                                 "-1: do not use nms. [0, 1]")
        self.parser = parser

    def _get_default_v_feat_dirs(self, dset_name, saved_options):
        """Get default relative paths for video features based on dataset name."""
        # Map dataset names to relative feature directory structures
        feature_path_map = {
            "hl": ["../Datasets/hl/features/internvideo2_video_features",
                   "../Datasets/hl/features/clip_video_features",
                   "../Datasets/hl/features/blip_video_features"],
            "qvhl": ["../Datasets/qvhl/features/internvideo2_video_features",
                     "../Datasets/qvhl/features/clip_video_features",
                     "../Datasets/qvhl/features/blip_video_features"],
            "charadesSTA": ["../Datasets/charades/internvideo2_video_features",
                            "../Datasets/charades/clip_video_features",
                            "../Datasets/charades/blip_video_features"],
            "tacos": ["../Datasets/tacos/internvideo2_video_features",
                      "../Datasets/tacos/clip_video_features",
                      "../Datasets/tacos/blip_video_features"],
    
        }
        return feature_path_map.get(dset_name, saved_options.get("v_feat_dirs", []))

    def _get_default_t_feat_dirs(self, dset_name, saved_options):
        """Get default relative paths for text features based on dataset name."""
        # Map dataset names to relative feature directory structures
        feature_path_map = {
            "hl": ["../Datasets/hl/features/clip_text_features",
                   "../Datasets/hl/features/blip_aug_text_features_openai"],
            "qvhl": ["../Datasets/qvhl/features/clip_text_features",
                     "../Datasets/qvhl/features/blip_aug_text_features_openai"],
            "charadesSTA": ["../Datasets/charades/clip_text_features",
                            "../Datasets/charades/blip_text_features"],
            "tacos": ["../Datasets/tacos/clip_text_features",
                      "../Datasets/tacos/blip_text_features"],
          
        }
        return feature_path_map.get(dset_name, saved_options.get("t_feat_dirs", []))

    def display_save(self, opt):
        args = vars(opt)
        # Display settings
        print(dict_to_markdown(vars(opt), max_str_len=120))
        # Save settings
        if not isinstance(self, TestOptions):
            option_file_path = os.path.join(opt.results_dir, self.saved_option_filename)  # not yaml file indeed
            save_json(args, option_file_path, save_pretty=True)

    def parse(self, a_feat_dir=None):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()

        if opt.debug:
            opt.results_root = os.path.sep.join(opt.results_root.split(os.path.sep)[:-1] + ["debug_results", ])
            opt.num_workers = 0

        if isinstance(self, TestOptions):
            # modify model_dir to absolute path
            # opt.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", opt.model_dir)
            opt.model_dir = os.path.dirname(opt.resume)
            if a_feat_dir is not None:
                opt.a_feat_dir = a_feat_dir
            
            # Try to load config from checkpoint first (merged checkpoint)
            saved_options = None
            if opt.resume is not None:
                try:
                    checkpoint = torch.load(opt.resume, map_location="cpu")
                    if 'config' in checkpoint:
                        saved_options = checkpoint['config']
                        # Apply backward compatibility conversions
                        if backward_compat_param_names(saved_options):
                            print(f"Applied backward compatibility conversions to checkpoint config")
                        print(f"Loaded config from checkpoint: {opt.resume}")
                except Exception as e:
                    print(f"Could not load config from checkpoint: {e}")
            
            # Fall back to opt.json if config not in checkpoint
            if saved_options is None:
                opt_json_path = os.path.join(opt.model_dir, self.saved_option_filename)
                if os.path.exists(opt_json_path):
                    saved_options = load_json(opt_json_path)
                    # Apply backward compatibility conversions
                    if backward_compat_param_names(saved_options):
                        print(f"Applied backward compatibility conversions to opt.json")
                else:
                    raise FileNotFoundError(
                        f"Could not find config in checkpoint or at {opt_json_path}. "
                        f"Please either: 1) Use a merged checkpoint (see tools/merge_checkpoint_with_config.py), "
                        f"or 2) Ensure opt.json exists in the checkpoint directory."
                    )
            
            for arg in saved_options:  # use saved options to overwrite all BaseOptions args.
                if arg not in ["results_root", "num_workers", "nms_thd", "debug",  # "max_before_nms", "max_after_nms"
                               "max_pred_l", "min_pred_l", 'eval_split_name',
                               "resume", "resume_all", "no_sort_results", "output_dir"]:
                    setattr(opt, arg, saved_options[arg])
            # opt.no_core_driver = True
            if opt.eval_results_dir is not None:
                opt.results_dir = opt.eval_results_dir
            else:
                # Use output_dir if specified, otherwise use current directory
                opt.results_dir = opt.output_dir if opt.output_dir else "."
        else:
            if opt.exp_id is None:
                raise ValueError("--exp_id is required for at a training option!")

            ctx_str = opt.ctx_mode + "_sub" if any(["sub_ctx" in p for p in opt.v_feat_dirs]) else opt.ctx_mode
            opt.results_dir = os.path.join(opt.results_root,
                                           "-".join([opt.dset_name, ctx_str, opt.exp_id,
                                                     time.strftime("%Y_%m_%d_%H_%M_%S")]))
            mkdirp(opt.results_dir)
            # save a copy of current code
            code_dir = os.path.dirname(os.path.realpath(__file__))
            code_zip_filename = os.path.join(opt.results_dir, "code.zip")
            make_zipfile(code_dir, code_zip_filename,
                         enclosing_dir="code",
                         exclude_dirs_substring="results",
                         exclude_dirs=["results", "debug_results", "__pycache__"],
                         exclude_extensions=[".pyc", ".ipynb", ".swap"], )

        self.display_save(opt)

        opt.ckpt_filepath = os.path.join(opt.results_dir, self.ckpt_filename)
        opt.train_log_filepath = os.path.join(opt.results_dir, self.train_log_filename)
        opt.eval_log_filepath = os.path.join(opt.results_dir, self.eval_log_filename)
        opt.tensorboard_log_dir = os.path.join(opt.results_dir, self.tensorboard_log_dir)
        opt.device = torch.device("cuda" if opt.device >= 0 else "cpu")
        opt.pin_memory = not opt.no_pin_memory

        opt.use_tef = "tef" in opt.ctx_mode
        opt.use_video = "video" in opt.ctx_mode
        if not opt.use_video:
            opt.v_feat_dim = 0
        if opt.use_tef:
            opt.v_feat_dim += 2

        self.opt = opt
        return opt


class TestOptions(BaseOptions):
    """add additional options for evaluating"""

    def initialize(self):
        BaseOptions.initialize(self)
        # also need to specify --eval_split_name
        self.parser.add_argument("--eval_id", type=str, help="evaluation id")
        self.parser.add_argument("--eval_results_dir", type=str, default=None,
                                 help="dir to save results, if not set, fall back to training results_dir")
        self.parser.add_argument("--model_dir", type=str,
                                 help="dir contains the model file, will be converted to absolute path afterwards")
        self.parser.add_argument("--output_dir", type=str, default=".",
                                 help="directory to save inference results")
