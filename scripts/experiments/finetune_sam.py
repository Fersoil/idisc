#!/usr/bin/env python
"""
Fine-tune iDisc with SAM3 queries.
Freezes pixel_encoder, pixel_decoder, and AFP.
Only trains: sam3_proj (Linear(256,128) per resolution) + ISD heads.

Modes:
  --mode concat   Concatenate SAM3 IDRs with AFP (default)
  --mode replace  Replace AFP entirely with SAM3 IDRs

Query source:
  --sam3-cache-dir  Use pre-cached video queries from dataloader (F4)
  --sam-checkpoint  Run SAM3 online per image (F1/F2/F3)
"""

import argparse
import json
import os
import random
from time import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data.distributed
from torch import nn, optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, SequentialSampler

import idisc.dataloders as custom_dataset
from idisc.models.idisc import IDisc
from idisc.utils import (DICT_METRICS_DEPTH, RunningMetric, format_seconds,
                         is_main_process)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
MULTI_CLASS_PROMPT = "car . truck . person . bicycle . building . tree . road sign . pole"
KITTI_CLASSES = ["car", "truck", "person", "bicycle", "building", "tree", "road sign", "pole"]


def denormalize_imagenet(img_tensor):
    mean = IMAGENET_MEAN.to(img_tensor.device)
    std = IMAGENET_STD.to(img_tensor.device)
    img = img_tensor * std + mean
    return (img * 255).clamp(0, 255).byte()


def extract_online_queries(proc, raw_img, prompt_mode):
    """Extract SAM3 instance queries via online inference."""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = proc.set_image(raw_img)
        if prompt_mode == "singleclass":
            all_queries, all_scores = [], []
            for cls in KITTI_CLASSES:
                proc.reset_all_prompts(state)
                proc.set_text_prompt(prompt=cls, state=state)
                iq = state.get("instance_queries")
                tk = state.get("topk_scores")
                if iq is not None and iq.shape[0] > 0:
                    all_queries.append(iq)
                    all_scores.append(tk)
            if not all_queries:
                return None
            all_queries = torch.cat(all_queries, dim=0)
            all_scores = torch.cat(all_scores, dim=0)
            top_k = min(proc.top_k_queries, all_queries.shape[0])
            _, best_idx = all_scores.topk(top_k)
            return all_queries[best_idx].float().clone()
        else:
            prompt = MULTI_CLASS_PROMPT if prompt_mode == "multiclass" else ""
            proc.set_text_prompt(prompt=prompt, state=state)
            iq = state.get("instance_queries")
            return iq.float().clone() if iq is not None else None


def _flatten_sequence_batch(batch, device):
    """Flatten (B, T, ...) sequence batches into (B*T, ...) so a single per-frame
    training/validation loop handles both sequence and single-frame datasets."""
    images = batch["images"].to(device)
    B, T = images.shape[:2]
    data = images.view(B * T, *images.shape[2:])
    gt = batch["depths"].to(device).view(B * T, *batch["depths"].shape[2:])
    mask = batch["masks"].to(device).view(B * T, *batch["masks"].shape[2:])
    q = batch.get("sam3_queries")
    sam3_q = (
        q.to(device).view(B * T, q.shape[2], q.shape[3])
        if (q is not None and q.dim() == 4)
        else None
    )
    return data, gt, mask, sam3_q, B * T


def _unpack_batch(batch, device):
    """Returns (data, gt, mask, sam3_q, n_samples) for both single-frame and sequence batches."""
    if "images" in batch:
        return _flatten_sequence_batch(batch, device)
    data = batch["image"].to(device)
    gt = batch["gt"].to(device)
    mask = batch["mask"].to(device)
    q = batch.get("sam3_queries")
    sam3_q = q.to(device) if (q is not None and q.dim() >= 2) else None
    return data, gt, mask, sam3_q, data.shape[0]


def validate_model(model, valid_loader, context, device, sam_mode="concat",
                    sam_proc=None, prompt_mode=None):
    metrics_tracker = RunningMetric(list(DICT_METRICS_DEPTH.keys()))

    for i, batch in enumerate(valid_loader):
        data, gt, mask, sam3_q, _ = _unpack_batch(batch, device)

        batch_preds = []
        for idx in range(data.shape[0]):
            iq = None
            if sam_proc is not None:
                raw_img = denormalize_imagenet(data[idx])
                iq = extract_online_queries(sam_proc, raw_img, prompt_mode)
                if iq is not None:
                    iq = iq.to(device)
            elif sam3_q is not None:
                q = sam3_q[idx]
                if q.dim() >= 2 and q.shape[0] > 0:
                    iq = q

            with context:
                pred, _, _ = model(
                    data[idx:idx + 1],
                    instance_queries=iq,
                    sam_mode=sam_mode,
                    gt=gt[idx:idx + 1],
                    mask=mask[idx:idx + 1],
                )
            batch_preds.append(pred)

        preds = torch.cat(batch_preds, dim=0)
        metrics_tracker.accumulate_metrics(
            gt.permute(0, 2, 3, 1),
            preds.permute(0, 2, 3, 1),
            mask.permute(0, 2, 3, 1),
        )
        if (i + 1) % 100 == 0:
            print(f"  Val: {i+1}/{len(valid_loader)}", flush=True)

    metrics = metrics_tracker.get_metrics()
    metrics_tracker.reset_metrics()
    return metrics


def _resolve_finetune_mode(cfg: dict[str, Any]) -> str:
    mode = cfg.get("mode")
    if mode in {"concat", "replace", "translate"}:
        return mode

    variant = cfg.get("variant")
    if variant == "sam-replace":
        return "replace"
    if variant == "sam-translate":
        return "translate"
    if variant in {"sam-concat", "sam-cached-video", "concat", "replace"}:
        return "concat" if variant != "replace" else "replace"

    raise ValueError(f"Cannot infer finetune mode from variant={variant!r}")


def run_finetune(cfg: dict[str, Any]) -> dict[str, Any]:
    config = cfg.get("config")
    if config is None:
        with open(cfg["config_file"], "r", encoding="utf-8") as f:
            config = json.load(f)

    sam_mode = _resolve_finetune_mode(cfg)
    prompt_mode = cfg.get("prompt_mode", "multiclass")
    sam_checkpoint = cfg.get("sam_checkpoint")
    sam3_cache_dir = cfg.get("sam3_cache_dir")

    encoder_name = config["model"]["pixel_encoder"].get("name", "")
    encoder_owns_sam3 = encoder_name in ("sam3_image", "sam3_video")
    encoder_is_video = encoder_name == "sam3_video"

    if not encoder_owns_sam3 and sam_checkpoint is None and sam3_cache_dir is None:
        raise ValueError("Either sam_checkpoint or sam3_cache_dir is required")

    if cfg.get("use_sequence_dataset"):
        config["data"]["train_dataset"] = "KITTISequenceDataset"

    finetune_cfg = cfg.get("finetune", {})
    output_dir = finetune_cfg.get("output_dir", cfg.get("output_dir", "finetune_output"))
    n_iters = int(finetune_cfg.get("n_iters", cfg.get("n_iters", 5000)))
    lr = float(finetune_cfg.get("lr", cfg.get("lr", 5e-5)))
    val_interval = int(finetune_cfg.get("val_interval", cfg.get("val_interval", 500)))
    batch_size = int(finetune_cfg.get("batch_size", cfg.get("batch_size", 2)))

    use_online_sam = sam_checkpoint is not None and not encoder_owns_sam3

    seed = config["generic"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Device: {device}", flush=True)
    print(f"Mode: {sam_mode}", flush=True)
    if use_online_sam:
        print(f"Prompt: {prompt_mode}", flush=True)
    else:
        print(f"Cache: {sam3_cache_dir}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # Load iDisc
    print("Loading iDisc model...", flush=True)
    model = IDisc.build(config).to(device)
    load_pretrained = cfg.get("load_pretrained", not encoder_owns_sam3)
    if load_pretrained:
        model.load_pretrained(cfg["model_file"])
        print("  iDisc loaded (pretrained weights).", flush=True)
    else:
        print("  iDisc built from scratch (no pretrained weights).", flush=True)

    if encoder_owns_sam3:
        # Pure SAM3 + d2c: SAM3 is frozen inside the encoder; iDisc-side
        # modules (pixel_decoder, AFP, ISD, sam3_proj) train fresh.
        for param in model.parameters():
            param.requires_grad = True
        # Freeze the SAM3 weights (image encoder or video encoder).
        sam_model = getattr(model.pixel_encoder, "sam_model",
                            getattr(model.pixel_encoder, "video_model", None))
        if sam_model is not None:
            for param in sam_model.parameters():
                param.requires_grad = False
    else:
        # Legacy fine-tune: freeze everything except sam3_proj and ISD.
        for param in model.parameters():
            param.requires_grad = False
        for param in model.sam3_proj.parameters():
            param.requires_grad = True
        for param in model.isd.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)", flush=True)

    # Datasets
    cache_dir = sam3_cache_dir if not use_online_sam else None
    data_path = os.path.join(cfg["base_path"], config["data"]["data_root"])
    print(f"Loading data from {data_path}...", flush=True)
    train_dataset = getattr(custom_dataset, config["data"]["train_dataset"])(
        test_mode=False,
        base_path=data_path,
        crop=config["data"].get("crop"),
        augmentations_db=config["data"].get("augmentations", {}),
        sam3_cache_dir=cache_dir,
        manifest_path=config["data"].get("manifest_path"),
        clip_length=config["data"].get("clip_length", 4),
        stride=config["data"].get("stride"),
    )
    valid_dataset = getattr(custom_dataset, config["data"]["val_dataset"])(
        test_mode=True,
        base_path=data_path,
        crop=config["data"].get("crop"),
        sam3_cache_dir=cache_dir,
        manifest_path=config["data"].get("manifest_path"),
        clip_length=config["data"].get("clip_length", 4),
        stride=config["data"].get("stride"),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=1, shuffle=False,
        num_workers=2, sampler=SequentialSampler(valid_dataset),
        pin_memory=True, drop_last=False,
    )
    print(f"  Train: {len(train_dataset)}, Val: {len(valid_dataset)}", flush=True)

    # Load SAM3 for online inference (F1/F2/F3)
    sam_proc = None
    if use_online_sam:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        sam_model = build_sam3_image_model(
            device=str(device), checkpoint_path=sam_checkpoint, load_from_HF=False)
        sam_model.eval()
        sam_proc = Sam3Processor(sam_model, device=str(device), use_presence_score=True)
        print("  SAM3 loaded for online inference.", flush=True)

    # Optimizer (only trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    scheduler = OneCycleLR(
        optimizer, max_lr=lr, total_steps=n_iters,
        pct_start=0.1, div_factor=10, final_div_factor=100,
    )

    f16 = config["training"].get("f16", False)
    context = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=f16)
    best_abs_rel = np.inf

    print(f"\nStart fine-tuning for {n_iters} iterations (lr={lr})...", flush=True)
    start = time()
    step = 0
    model.train()

    while step < n_iters:
        for batch in train_loader:
            if step >= n_iters:
                break

            optimizer.zero_grad()
            total_loss = 0.0
            valid_frames = 0

            if encoder_is_video and "images" in batch:
                # Video-encoder path: pass each clip in the batch to the
                # encoder, get per-frame (FPN, queries), then run iDisc
                # per frame.  Batch is (B, T, 3, H, W); no flatten needed.
                images = batch["images"].to(device)  # (B, T, 3, H, W)
                depths = batch["depths"].to(device)  # (B, T, 1, H, W)
                masks  = batch["masks"].to(device)   # (B, T, 1, H, W)
                B, T = images.shape[:2]
                n_samples = B * T

                for b in range(B):
                    clip = images[b]   # (T, 3, H, W)
                    # encoder.forward(clip) → (*FPN_T, queries_T)
                    enc_out = model.pixel_encoder(clip)
                    # Last element is queries (T, K, 256); rest are FPN levels
                    fpn_levels = enc_out[:-1]      # each (T, C, h, w)
                    queries_T  = enc_out[-1]       # (T, K, 256)

                    for t in range(T):
                        gt_t   = depths[b, t : t + 1]     # (1, 1, H, W)
                        mask_t = masks[b, t : t + 1]      # (1, 1, H, W)
                        if mask_t.bool().sum() == 0:
                            continue

                        # Provide the pre-extracted single-frame FPN to iDisc.
                        frame_fpn = tuple(
                            lvl[t : t + 1] for lvl in fpn_levels
                        )
                        # iDisc.forward needs the FPN in encoder output order
                        # (high-res first); invert_encoder_output_order then
                        # pixel_decoder handle the rest.  Pre-extracted FPN
                        # is in iDisc encoder_output order (already inverted);
                        # pass via pre_extracted_encoder_outputs after
                        # re-inverting so the model inverts it back.
                        inv_frame_fpn = tuple(reversed(frame_fpn))

                        iq = queries_T[t]   # (K, 256)

                        with context:
                            pred, losses, _ = model(
                                images[b, t : t + 1],  # still needed for shape
                                instance_queries=iq,
                                sam_mode=sam_mode,
                                pre_extracted_encoder_outputs=inv_frame_fpn,
                                gt=gt_t,
                                mask=mask_t,
                            )
                            loss = sum(v for v in losses["opt"].values()) / n_samples

                        loss.backward()
                        total_loss += loss.item()
                        valid_frames += 1
            else:
                # Standard (image-encoder) path.
                data, gt, mask, sam3_q, n_samples = _unpack_batch(batch, device)

                for idx in range(n_samples):
                    # Skip frames whose GT mask is all-False (sequence dataset
                    # zero-fills GT for frames without LiDAR labels; SILog over
                    # zero valid pixels returns NaN and poisons the model).
                    if mask[idx].bool().sum() == 0:
                        continue

                    iq = None
                    if use_online_sam:
                        raw_img = denormalize_imagenet(data[idx])
                        with torch.no_grad():
                            iq = extract_online_queries(sam_proc, raw_img, prompt_mode)
                        if iq is not None:
                            iq = iq.to(device)
                    elif sam3_q is not None:
                        q = sam3_q[idx]
                        if q.dim() >= 2 and q.shape[0] > 0:
                            iq = q

                    with context:
                        pred, losses, _ = model(
                            data[idx:idx + 1],
                            instance_queries=iq,
                            sam_mode=sam_mode,
                            gt=gt[idx:idx + 1],
                            mask=mask[idx:idx + 1],
                        )
                        loss = sum(v for v in losses["opt"].values()) / n_samples

                    loss.backward()
                    total_loss += loss.item()
                    valid_frames += 1

            if valid_frames == 0:
                continue

            nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % 100 == 0:
                elapsed = int(time() - start)
                eta = int(elapsed * (n_iters - step) / max(1, step))
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"  Step {step}/{n_iters} | loss={total_loss:.5f} | "
                    f"lr={lr_now:.2e} | [{format_seconds(elapsed)}<{format_seconds(eta)}]",
                    flush=True,
                )

            if step % val_interval == 0 or step == n_iters:
                print(f"\n  Validation at step {step}...", flush=True)
                model.eval()
                with torch.no_grad():
                    metrics = validate_model(model, valid_loader, context, device,
                                             sam_mode=sam_mode, sam_proc=sam_proc,
                                             prompt_mode=prompt_mode if use_online_sam else None)
                model.train()
                if encoder_owns_sam3:
                    for param in model.parameters():
                        param.requires_grad = True
                    sam_model = getattr(model.pixel_encoder, "sam_model",
                                        getattr(model.pixel_encoder, "video_model", None))
                    if sam_model is not None:
                        for param in sam_model.parameters():
                            param.requires_grad = False
                else:
                    for param in model.parameters():
                        param.requires_grad = False
                    for param in model.sam3_proj.parameters():
                        param.requires_grad = True
                    for param in model.isd.parameters():
                        param.requires_grad = True

                abs_rel = metrics.get("abs_rel", np.inf)
                print(f"  abs_rel={abs_rel:.6f}  (best={best_abs_rel:.6f})", flush=True)
                for k, v in sorted(metrics.items()):
                    print(f"    {k}: {v:.6f}", flush=True)

                # Drop frozen SAM3 weights from checkpoints — they're reloaded
                # from sam_checkpoint at startup, so persisting them per-save
                # bloats the checkpoint by ~16x and triggers OOM/EDQUOT.
                def _trainable_state_dict():
                    sd = model.state_dict()
                    if encoder_owns_sam3:
                        sd = {k: v for k, v in sd.items()
                              if not k.startswith("pixel_encoder.sam_model.")
                              and not k.startswith("pixel_encoder.video_model.")}
                    return sd

                if abs_rel < best_abs_rel:
                    best_abs_rel = abs_rel
                    ckpt_path = os.path.join(output_dir, "best_sam_finetuned.pt")
                    torch.save(_trainable_state_dict(), ckpt_path)
                    print(f"  New best! Saved to {ckpt_path}", flush=True)

                ckpt_path = os.path.join(output_dir, f"checkpoint_step{step}.pt")
                torch.save(_trainable_state_dict(), ckpt_path)
                print(f"  Checkpoint saved to {ckpt_path}\n", flush=True)

    # Final save
    final_path = os.path.join(output_dir, "final_sam_finetuned.pt")
    final_sd = model.state_dict()
    if encoder_owns_sam3:
        final_sd = {k: v for k, v in final_sd.items()
                    if not k.startswith("pixel_encoder.sam_model.")
                    and not k.startswith("pixel_encoder.video_model.")}
    torch.save(final_sd, final_path)
    print(f"\nFine-tuning complete. Final model saved to {final_path}", flush=True)
    print(f"Best abs_rel: {best_abs_rel:.6f}", flush=True)

    return {
        "best_abs_rel": float(best_abs_rel),
        "final_checkpoint": final_path,
        "output_dir": output_dir,
        "mode": sam_mode,
        "prompt_mode": prompt_mode,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune iDisc with SAM3 queries")
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--model-file", type=str, required=True)
    parser.add_argument("--base-path", type=str, required=True)
    parser.add_argument("--mode", type=str, default="concat", choices=["concat", "replace"],
                        help="concat: SAM3 IDRs + AFP; replace: SAM3 IDRs only")
    parser.add_argument("--prompt-mode", type=str, default="multiclass",
                        choices=["multiclass", "singleclass"],
                        help="Text prompt strategy for online SAM3")
    parser.add_argument("--sam-checkpoint", type=str, default=None,
                        help="SAM3 checkpoint for online inference (F1/F2/F3)")
    parser.add_argument("--sam3-cache-dir", type=str, default=None,
                        help="Directory with pre-cached SAM3 video queries (F4)")
    parser.add_argument("--output-dir", type=str, default="finetune_output")
    parser.add_argument("--n-iters", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--val-interval", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--use-sequence-dataset", action="store_true",
                        help="Override config to use KITTISequenceDataset for training")
    return parser.parse_args()


def main():
    args = _parse_args()
    run_finetune(
        {
            "config_file": args.config_file,
            "model_file": args.model_file,
            "base_path": args.base_path,
            "mode": args.mode,
            "prompt_mode": args.prompt_mode,
            "sam_checkpoint": args.sam_checkpoint,
            "sam3_cache_dir": args.sam3_cache_dir,
            "output_dir": args.output_dir,
            "n_iters": args.n_iters,
            "lr": args.lr,
            "val_interval": args.val_interval,
            "batch_size": args.batch_size,
            "use_sequence_dataset": args.use_sequence_dataset,
        }
    )


if __name__ == "__main__":
    main()
