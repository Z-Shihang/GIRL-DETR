import sys
import os
# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pprint
from tqdm import tqdm, trange
import numpy as np
from collections import OrderedDict, defaultdict
from utils.basic_utils import AverageMeter

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from girl.config import TestOptions
from girl.model import build_model
from girl.span_utils import span_cxw_to_xx
from girl.start_end_dataset import StartEndDataset, start_end_collate, prepare_batch_inputs
# from qd_detr.start_end_dataset_audio import \
#     StartEndDataset_audio, start_end_collate_audio, prepare_batch_inputs_audio
from girl.postprocessing import PostProcessorDETR
from standalone_eval.eval import eval_submission
from utils.basic_utils import save_jsonl, save_json
from utils.temporal_nms import temporal_nms

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def build_finetune_prefixes(opt):
    """Return trainable module prefixes for frozen-backbone SCST experiments."""
    head_prefixes = {"span_embed", "class_embed", "iou_embed"}
    extra_prefixes = set()
    group_names = ["prediction_heads"]

    if getattr(opt, "unfreeze_query_embed", False):
        extra_prefixes.add("query_embed")
        group_names.append("query_embed")

    if getattr(opt, "unfreeze_decoder_geometry", False):
        extra_prefixes.update({
            "transformer.decoder.bbox_embed",
            "transformer.decoder.ref_point_head",
            "transformer.decoder.query_scale",
            "transformer.decoder.ref_anchor_head",
        })
        group_names.append("decoder_geometry")

    if getattr(opt, "unfreeze_decoder_gate", False):
        extra_prefixes.update({
            "transformer.decoder.gate_cross_attn",
            "transformer.decoder.gate_self_attn",
            "transformer.decoder.gate_linear",
            "transformer.decoder.gate_norm",
        })
        group_names.append("decoder_gate")

    if getattr(opt, "unfreeze_global_tokens", False):
        extra_prefixes.update({"global_rep_token", "global_rep_pos"})
        group_names.append("global_tokens")

    if getattr(opt, "unfreeze_input_proj", False):
        extra_prefixes.update({"input_vid_proj", "input_txt_proj"})
        group_names.append("input_proj")

    if getattr(opt, "unfreeze_feature_refinement", False):
        extra_prefixes.add("feature_refinement")
        group_names.append("feature_refinement")

    if getattr(opt, "unfreeze_saliency_proj", False):
        extra_prefixes.update({"saliency_proj1", "saliency_proj2", "saliency_proj_mr"})
        group_names.append("saliency_proj")

    if getattr(opt, "unfreeze_cmi", False):
        extra_prefixes.add("transformer.t2v_encoder")
        group_names.append("cmi")

    if getattr(opt, "unfreeze_encoder_last", False):
        enc_layers = getattr(opt, "enc_layers", 3)
        extra_prefixes.add(f"transformer.encoder.layers.{enc_layers - 1}")
        group_names.append("encoder_last")

    if getattr(opt, "unfreeze_decoder_last", False):
        dec_layers = getattr(opt, "dec_layers", 3)
        extra_prefixes.add(f"transformer.decoder.layers.{dec_layers - 1}")
        group_names.append("decoder_last")

    if getattr(opt, "unfreeze_decoder_all", False):
        dec_layers = getattr(opt, "dec_layers", 3)
        for i in range(dec_layers):
            extra_prefixes.add(f"transformer.decoder.layers.{i}")
        extra_prefixes.add("transformer.decoder.norm")
        group_names.append("decoder_all")

    return head_prefixes, extra_prefixes, group_names


def matches_prefix(name, prefixes):
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)


def configure_frozen_backbone(model, opt):
    """Apply frozen-backbone policy before optimizer creation."""
    if not getattr(opt, "freeze_backbone", False):
        return

    head_prefixes, extra_prefixes, group_names = build_finetune_prefixes(opt)
    trainable_prefixes = head_prefixes | extra_prefixes
    frozen_count, trainable_count = 0, 0
    for name, param in model.named_parameters():
        if matches_prefix(name, trainable_prefixes):
            param.requires_grad = True
            trainable_count += 1
        else:
            param.requires_grad = False
            frozen_count += 1

    trainable_numel = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"[Freeze Backbone] Frozen {frozen_count} tensors, trainable {trainable_count} tensors "
        f"({trainable_numel:,} scalars). Groups: {', '.join(group_names)}"
    )
    logger.info(f"[Freeze Backbone] Trainable head prefixes: {', '.join(sorted(head_prefixes))}")
    if extra_prefixes:
        logger.info(f"[Freeze Backbone] Trainable extra prefixes: {', '.join(sorted(extra_prefixes))}")


def post_processing_mr_nms(mr_res, nms_thd, max_before_nms, max_after_nms):
    mr_res_after_nms = []
    for e in mr_res:
        e["pred_relevant_windows"] = temporal_nms(
            e["pred_relevant_windows"][:max_before_nms],
            nms_thd=nms_thd,
            max_after_nms=max_after_nms
        )
        mr_res_after_nms.append(e)
    return mr_res_after_nms


def eval_epoch_post_processing(submission, opt, gt_data, save_submission_filename):
    # IOU_THDS = (0.5, 0.7)
    logger.info("Saving/Evaluating before nms results")
    submission_path = os.path.join(opt.results_dir, save_submission_filename)
    save_jsonl(submission, submission_path)

    if opt.eval_split_name in ["val"]:  # since test_public has no GT
        metrics = eval_submission(
            submission, gt_data,
            verbose=opt.debug, match_number=not opt.debug
        )
        save_metrics_path = submission_path.replace(".jsonl", "_metrics.json")
        save_json(metrics, save_metrics_path, save_pretty=True, sort_keys=False)
        latest_file_paths = [submission_path, save_metrics_path]
    else:
        metrics = None
        latest_file_paths = [submission_path, ]

    if opt.nms_thd != -1:
        logger.info("[MR] Performing nms with nms_thd {}".format(opt.nms_thd))
        submission_after_nms = post_processing_mr_nms(
            submission, nms_thd=opt.nms_thd,
            max_before_nms=opt.max_before_nms, max_after_nms=opt.max_after_nms
        )

        logger.info("Saving/Evaluating nms results")
        submission_nms_path = submission_path.replace(".jsonl", "_nms_thd_{}.jsonl".format(opt.nms_thd))
        save_jsonl(submission_after_nms, submission_nms_path)
        if opt.eval_split_name == "val":
            metrics_nms = eval_submission(
                submission_after_nms, gt_data,
                verbose=opt.debug, match_number=not opt.debug
            )
            save_metrics_nms_path = submission_nms_path.replace(".jsonl", "_metrics.json")
            save_json(metrics_nms, save_metrics_nms_path, save_pretty=True, sort_keys=False)
            latest_file_paths += [submission_nms_path, save_metrics_nms_path]
        else:
            metrics_nms = None
            latest_file_paths = [submission_nms_path, ]
    else:
        metrics_nms = None
    return metrics, metrics_nms, latest_file_paths


# for HL
@torch.no_grad()
def compute_hl_results(model, eval_loader, opt, epoch_i=None, criterion=None, tb_writer=None):
    model.eval()
    if criterion:
        assert eval_loader.dataset.load_labels
        criterion.eval()

    loss_meters = defaultdict(AverageMeter)
    write_tb = tb_writer is not None and epoch_i is not None

    mr_res = []

    topk = 5  # top-5 map

    video_ap_collected = []
    for batch in tqdm(eval_loader, desc="compute st ed scores"):
        query_meta = batch[0]

        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)

        outputs = model(**model_inputs)

        # loss meters
        # if criterion:
        #     loss_dict = criterion(outputs, targets)
        #     weight_dict = criterion.weight_dict
        # print(loss_dict)
        # print(weight_dict)
        # print('#######')
        # {'loss_saliency': tensor(18.1374, device='cuda:0')}
        # {'loss_span': 10, 'loss_giou': 1, 'loss_label': 4, 'loss_saliency': 1.0, 'loss_ms_align': 1.0,
        #  'loss_distill': 1.0, 'loss_span_0': 10, 'loss_giou_0': 1, 'loss_label_0': 4, 'loss_ms_align_0': 1.0,
        #  'loss_distill_0': 1.0}
        # losses=0.
        # print(loss_dict.keys(), weight_dict.keys())
        # losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # loss_dict["loss_overall"] = float(losses)  # for logging only
        # print(loss_dict.items())
        #
        # print(weight_dict.items())
        # for k, v in loss_dict.items():
        #     loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

        preds = outputs['saliency_scores'].clone().detach()

        for meta, pred in zip(query_meta, preds):
            pred = pred
            label = meta['label']  # raw label

            video_ap = []
            # Follow the UMT code "https://github.com/TencentARC/UMT/blob/main/datasets/tvsum.py"

            if opt.dset_name in ["tvsum"]:
                for i in range(20):
                    pred = pred.cpu()
                    cur_pred = pred[:len(label)]
                    inds = torch.argsort(cur_pred, descending=True, dim=-1)

                    # video_id = self.get_video_id(idx)
                    cur_label = torch.Tensor(label)[:, i]
                    cur_label = torch.where(cur_label > cur_label.median(), 1.0, .0)

                    cur_label = cur_label[inds].tolist()[:topk]

                    # if (num_gt := sum(cur_label)) == 0:
                    num_gt = sum(cur_label)
                    if num_gt == 0:
                        video_ap.append(0)
                        continue

                    hits = ap = rec = 0
                    prc = 1

                    for j, gt in enumerate(cur_label):
                        hits += gt

                        _rec = hits / num_gt
                        _prc = hits / (j + 1)

                        ap += (_rec - rec) * (prc + _prc) / 2
                        rec, prc = _rec, _prc

                    video_ap.append(ap)

            elif opt.dset_name in ["youtube_uni"]:
                cur_pred = pred[:len(label)]
                # if opt.dset_name == "tvsum_sfc":
                cur_pred = cur_pred.cpu()
                inds = torch.argsort(cur_pred, descending=True, dim=-1)

                cur_label = torch.Tensor(label).squeeze()[inds].tolist()

                num_gt = sum(cur_label)
                if num_gt == 0:
                    video_ap.append(0)
                    continue

                hits = ap = rec = 0
                prc = 1

                for j, gt in enumerate(cur_label):
                    hits += gt

                    _rec = hits / num_gt
                    _prc = hits / (j + 1)

                    ap += (_rec - rec) * (prc + _prc) / 2
                    rec, prc = _rec, _prc

                video_ap.append(float(ap))
            else:
                print("No such dataset")
                exit(-1)

            video_ap_collected.append(video_ap)

    mean_ap = np.mean(video_ap_collected)
    submmission = dict(mAP=round(mean_ap, 5))

    # tensorboard writer
    if write_tb and criterion:
        for k, v in loss_meters.items():
            tb_writer.add_scalar("Eval/{}".format(k), v.avg, epoch_i + 1)

    return submmission, loss_meters



@torch.no_grad()
def compute_mr_results(model, eval_loader, opt, epoch_i=None, criterion=None, tb_writer=None):
    model.eval()
    if criterion:
        assert eval_loader.dataset.load_labels
        criterion.eval()

    loss_meters = defaultdict(AverageMeter)
    write_tb = tb_writer is not None and epoch_i is not None

    mr_res = []
    for batch in tqdm(eval_loader, desc="compute st ed scores"):
        query_meta = batch[0]
        # if opt.a_feat_dir is None:
        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)
        # else:
        #     model_inputs, targets = prepare_batch_inputs_audio(batch[1], opt.device, non_blocking=opt.pin_memory)
        outputs = model(**model_inputs)
        prob = F.softmax(outputs["pred_logits"], -1)  # (batch_size, #queries, #classes=2)
        if opt.span_loss_type == "l1":
            scores = prob[..., 0]  # * (batch_size, #queries)  foreground label is 0, we directly take it
            # IoU-aware scoring: modulate classification score with predicted IoU
            if 'pred_iou' in outputs and getattr(opt, 'use_iou_head', False):
                pred_iou = outputs['pred_iou']  # (bsz, #queries)
                scores = scores * pred_iou
            # IoU blended scoring: alpha * pred_iou + (1-alpha) * fg_score
            iou_alpha = getattr(opt, 'iou_rerank_alpha', 0.0)
            if iou_alpha > 0 and 'pred_iou' in outputs and not getattr(opt, 'use_iou_head', False):
                pred_iou = outputs['pred_iou'].clamp(0, 1)  # (bsz, #queries)
                scores = (1.0 - iou_alpha) * scores + iou_alpha * pred_iou
            pred_spans = outputs["pred_spans"]  # (bsz, #queries, 2)
            _saliency_scores = outputs["saliency_scores"].half()  # (bsz, L)
            saliency_scores = []
            valid_vid_lengths = model_inputs["src_vid_mask"].sum(1).cpu().tolist()
            for j in range(len(valid_vid_lengths)):
                saliency_scores.append(_saliency_scores[j, :int(valid_vid_lengths[j])].tolist())
        else:
            bsz, n_queries = outputs["pred_spans"].shape[:2]  # # (bsz, #queries, max_v_l *2)
            pred_spans_logits = outputs["pred_spans"].view(bsz, n_queries, 2, opt.max_v_l)
            # TODO use more advanced decoding method with st_ed product
            pred_span_scores, pred_spans = F.softmax(pred_spans_logits, dim=-1).max(-1)  # 2 * (bsz, #queries, 2)
            scores = torch.prod(pred_span_scores, 2)  # (bsz, #queries)
            pred_spans[:, 1] += 1
            pred_spans *= opt.clip_length

        # compose predictions
        for idx, (meta, spans, score) in enumerate(zip(query_meta, pred_spans.cpu(), scores.cpu())):
            if opt.span_loss_type == "l1":
                spans = span_cxw_to_xx(spans) * meta["duration"]
            # # (#queries, 3), [st(float), ed(float), score(float)]
            cur_ranked_preds = torch.cat([spans, score[:, None]], dim=1).tolist()
            if not opt.no_sort_results:
                cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
            cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]
            cur_query_pred = dict(
                qid=meta["qid"],
                query=meta["query"],
                vid=meta["vid"],
                pred_relevant_windows=cur_ranked_preds,
                pred_saliency_scores=saliency_scores[idx]
            )
            mr_res.append(cur_query_pred)

        if criterion:
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            loss_dict["loss_overall"] = float(losses)  # for logging only
            for k, v in loss_dict.items():
                loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

        if opt.debug:
            break

    if write_tb and criterion:
        for k, v in loss_meters.items():
            tb_writer.add_scalar("Eval/{}".format(k), v.avg, epoch_i + 1)

    if opt.dset_name in ['hl']:
        post_processor = PostProcessorDETR(
            clip_length=opt.clip_length, min_ts_val=0, max_ts_val=150,
            min_w_l=2, max_w_l=150, move_window_method="left",
            process_func_names=("clip_ts", "round_multiple")
        )
    elif opt.dset_name in ['charadesSTA']:
        if opt.v_feat_dim == 4096:  # vgg
            post_processor = PostProcessorDETR(
                clip_length=opt.clip_length, min_ts_val=0, max_ts_val=360,
                min_w_l=12, max_w_l=360, move_window_method="left",
                process_func_names=("clip_ts", "round_multiple")
            )
        else:
            post_processor = PostProcessorDETR(
                clip_length=opt.clip_length, min_ts_val=0, max_ts_val=150,
                min_w_l=2, max_w_l=60, move_window_method="left",
                process_func_names=("clip_ts", "round_multiple")
            )
    else:
        post_processor = PostProcessorDETR(
            clip_length=opt.clip_length, min_ts_val=0, max_ts_val=50000,
            min_w_l=0, max_w_l=50000, move_window_method="left",
            process_func_names=(["round_multiple"])
        )
    mr_res = post_processor(mr_res)
    return mr_res, loss_meters


def get_eval_res(model, eval_loader, opt, epoch_i, criterion, tb_writer):
    """compute and save query and video proposal embeddings"""
    eval_res, eval_loss_meters = compute_mr_results(model, eval_loader, opt, epoch_i, criterion, tb_writer)  # list(dict)
    return eval_res, eval_loss_meters


def eval_epoch(model, eval_dataset, opt, save_submission_filename, epoch_i=None, criterion=None, tb_writer=None):
    logger.info("Generate submissions")
    model.eval()
    if criterion is not None and eval_dataset.load_labels:
        criterion.eval()
    else:
        criterion = None

    # if opt.a_feat_dir is None:
    eval_loader = DataLoader(
        eval_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.eval_bsz,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=opt.pin_memory
    )
    # else:
    #     eval_loader = DataLoader(
    #         eval_dataset,
    #         collate_fn=start_end_collate_audio,
    #         batch_size=opt.eval_bsz,
    #         num_workers=opt.num_workers,
    #         shuffle=False,
    #         pin_memory=opt.pin_memory
    #     )

    # tvsum
    if opt.dset_name in ['tvsum']:
        metrics, eval_loss_meters = compute_hl_results(model, eval_loader, opt, epoch_i, criterion, tb_writer)

        # to match original save format
        submission = [
            {"brief": metrics}
        ]
        submission_path = os.path.join(opt.results_dir, "latest_metric.jsonl")
        save_jsonl(submission, submission_path)

        return submission[0], submission[0], eval_loss_meters, [submission_path]

    else:
        submission, eval_loss_meters = get_eval_res(model, eval_loader, opt, epoch_i, criterion, tb_writer)

        if opt.dset_name in ['charadesSTA', 'tacos', 'nlq']:
            new_submission = []
            for s in submission:
                s.pop('pred_saliency_scores', None)
                new_submission.append(s)
            submission = new_submission

        if opt.no_sort_results:
            save_submission_filename = save_submission_filename.replace(".jsonl", "_unsorted.jsonl")
        metrics, metrics_nms, latest_file_paths = eval_epoch_post_processing(
            submission, opt, eval_dataset.data, save_submission_filename)
        return metrics, metrics_nms, eval_loss_meters, latest_file_paths


def setup_model(opt):
    """setup model/optimizer/scheduler and load checkpoints when needed"""
    logger.info("setup model/optimizer/scheduler")
    model, criterion = build_model(opt)
    configure_frozen_backbone(model, opt)
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        criterion.to(opt.device)

    use_split_lr = (
        getattr(opt, "freeze_backbone", False)
        or getattr(opt, "lr_heads", None) is not None
        or getattr(opt, "lr_unfrozen", None) is not None
    )
    if use_split_lr:
        head_prefixes, _, _ = build_finetune_prefixes(opt)
        head_params = []
        extra_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if matches_prefix(name, head_prefixes):
                head_params.append(param)
            else:
                extra_params.append(param)

        lr_heads = opt.lr if getattr(opt, "lr_heads", None) is None else opt.lr_heads
        lr_unfrozen = opt.lr if getattr(opt, "lr_unfrozen", None) is None else opt.lr_unfrozen
        param_dicts = []
        if head_params:
            param_dicts.append({"params": head_params, "lr": lr_heads, "group_name": "prediction_heads"})
        if extra_params:
            param_dicts.append({"params": extra_params, "lr": lr_unfrozen, "group_name": "unfrozen_modules"})
        if not param_dicts:
            raise ValueError("No trainable parameters found after applying freeze/unfreeze settings.")

        group_log = ", ".join(
            f"{group['group_name']}={sum(p.numel() for p in group['params']):,}@lr={group['lr']}"
            for group in param_dicts
        )
        logger.info(f"[Optimizer] Parameter groups: {group_log}")
    else:
        param_dicts = [{"params": [p for _, p in model.named_parameters() if p.requires_grad], "group_name": "default"}]
        logger.info(
            f"[Optimizer] Parameter groups: default="
            f"{sum(p.numel() for p in param_dicts[0]['params']):,}@lr={opt.lr}"
        )
    optimizer = torch.optim.AdamW(param_dicts, lr=opt.lr, weight_decay=opt.wd)

    scheduler_type = getattr(opt, 'lr_scheduler', 'step')
    warmup_epochs = getattr(opt, 'warmup_epochs', 0)
    if scheduler_type == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opt.n_epoch - warmup_epochs, eta_min=opt.lr * 0.01)
    else:
        main_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_drop)

    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])
    else:
        lr_scheduler = main_scheduler

    if opt.resume is not None:
        logger.info(f"Load checkpoint from {opt.resume}")
        checkpoint = torch.load(opt.resume, map_location="cpu")
        missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
        if missing:
            logger.info(f"Missing keys (newly added): {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")
        if opt.resume_all:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                opt.start_epoch = checkpoint['epoch'] + 1
            except ValueError as exc:
                logger.warning(
                    f"resume_all skipped optimizer/scheduler state due to parameter-group mismatch: {exc}"
                )
        logger.info(f"Loaded model saved at epoch {checkpoint['epoch']} from checkpoint: {opt.resume}")
    else:
        logger.warning("If you intend to evaluate the model, please specify --resume with ckpt path")

    return model, criterion, optimizer, lr_scheduler


def start_inference(train_opt=None, split=None, splitfile=None):
    if train_opt is not None:
        opt = TestOptions().parse(train_opt.a_feat_dir)
    else:
        opt = TestOptions().parse()
    
    # If resume checkpoint contains config, load it
    if opt.resume is not None:
        try:
            checkpoint = torch.load(opt.resume, map_location="cpu")
            if 'config' in checkpoint:
                logger.info("Loading config from checkpoint...")
                import json
                from argparse import Namespace
                from girl.config import backward_compat_param_names
                config_dict = checkpoint['config']
                
                # Apply backward compatibility conversions
                if backward_compat_param_names(config_dict):
                    logger.info("Applied backward compatibility conversions to checkpoint config")
                
                # Merge checkpoint config with command-line options
                # Skip CLI-only arguments that should not be overwritten
                # v_feat_dim, t_feat_dim, a_feat_dim need to be calculated based on ctx_mode,
                # not directly loaded from config
                skip_keys = {"results_root", "num_workers", "nms_thd", "debug",
                            "max_pred_l", "min_pred_l", "eval_split_name", "device",
                            "v_feat_dim", "t_feat_dim", "a_feat_dim",
                            "resume", "resume_all", "no_sort_results", "output_dir"}
                for key, value in config_dict.items():
                    if key not in skip_keys:
                        setattr(opt, key, value)
        except Exception as e:
            logger.warning(f"Could not load config from checkpoint: {e}")
    
    if split is not None:
        opt.eval_split_name = split
    if splitfile is not None:
        opt.eval_path = splitfile

    print(opt.eval_split_name)
    print(opt.eval_path)
    logger.info("Setup config, data and model...")


    cudnn.benchmark = True
    cudnn.deterministic = False

    assert opt.eval_path is not None
    eval_dataset = StartEndDataset(
        dset_name=opt.dset_name,
        data_path=opt.eval_path,
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dirs=opt.t_feat_dirs,
        a_feat_dirs=opt.a_feat_dirs if opt.a_feat_dim > 0 else None,
        q_feat_type="last_hidden_state",
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        data_ratio=opt.data_ratio,
        normalize_v=not opt.no_norm_vfeat,
        normalize_t=not opt.no_norm_tfeat,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        load_labels=(opt.eval_split_name == "val"),
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=0,
        dset_domain=opt.dset_domain,
    )


    model, criterion, _, _ = setup_model(opt)

    # Ensure output directory exists
    if not os.path.exists(opt.results_dir):
        os.makedirs(opt.results_dir, exist_ok=True)
    
    save_submission_filename = "hl_{}_submission.jsonl".format(opt.eval_split_name)
    # save_submission_filename = "inference_{}_{}_{}_preds.jsonl".format(
    #     opt.dset_name, opt.eval_split_name, opt.eval_id)
    logger.info("Starting inference...")
    with torch.no_grad():
        metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = \
            eval_epoch(model, eval_dataset, opt, save_submission_filename, criterion=criterion)
    if opt.eval_split_name == 'val':
        logger.info("metrics_no_nms {}".format(pprint.pformat(metrics_no_nms["brief"], indent=4)))
    if metrics_nms is not None:
        logger.info("metrics_nms {}".format(pprint.pformat(metrics_nms["brief"], indent=4)))


if __name__ == '__main__':
    start_inference()
