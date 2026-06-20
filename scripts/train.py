#!/usr/bin/env python
"""Training backend for iDisc (baseline AFP) and SAM3+iDisc variants.

Entered from `run_with_hydra.py` after `build_runtime_config` has validated
the schema. Two execution paths share a single per-frame inner loop:

  - encoder=resnet101 (idr_source=afp)  → standard image dataloader, AFP IDRs.
  - encoder=sam3_image (idr_source=sam3) → SAM3 produces queries per frame;
        the linear_proj or adapter path consumes them; sequences get flattened.
  - encoder=sam3_video (idr_source=sam3) → FPN+queries pre-extracted per clip,
        then per-frame depth-head forward (avoids re-running the backbone).
"""

import os
import random
from time import time
from typing import Any

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.checkpoint import checkpoint

from scripts.experiments._eval_common import build_train_loader, build_val_loader
from idisc.models.idisc import IDisc
from idisc.models.lora import LoRALinear
from idisc.models.sam3_track import Sam3TrackModule
from idisc.optimization.grounding_losses import temporal_smoothness_loss
from idisc.utils import DICT_METRICS_DEPTH, RunningMetric, format_seconds
from idisc.utils.tracking import log_metrics

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _flatten_sequence_batch(batch, device):
    """Flatten (B, T, ...) sequence batches into (B*T, ...) so the per-frame
    training/validation loop handles both sequence and single-frame datasets."""
    images = batch["images"].to(device)
    B, T = images.shape[:2]
    data = images.view(B * T, *images.shape[2:])
    gt_key = "gt" if "gt" in batch else "depths"
    mask_key = "mask" if "mask" in batch else "masks"
    gt = batch[gt_key].to(device).view(B * T, *batch[gt_key].shape[2:])
    mask = batch[mask_key].to(device).view(B * T, *batch[mask_key].shape[2:])
    return data, gt, mask, B * T


def _unpack_batch(batch, device):
    if "images" in batch:
        return _flatten_sequence_batch(batch, device)
    data = batch["image"].to(device)
    gt = batch["gt"].to(device)
    mask = batch["mask"].to(device)
    return data, gt, mask, data.shape[0]


def _to_clip(batch, device):
    return (batch["images"].to(device), batch["depths"].to(device),
            batch["masks"].to(device))


def _valid_frames(masks):
    return int(masks.bool().any(dim=(2, 3, 4)).sum().item())


def _validate_image(model, valid_loader, context, device, sam_mode):
    tracker = RunningMetric(list(DICT_METRICS_DEPTH.keys()))
    for i, batch in enumerate(valid_loader):
        data, gt, mask, _ = _unpack_batch(batch, device)
        preds = []
        for idx in range(data.shape[0]):
            with context:
                pred, _, _ = model(
                    data[idx:idx + 1],
                    sam_mode=sam_mode,
                    gt=gt[idx:idx + 1],
                    mask=mask[idx:idx + 1],
                )
            preds.append(pred)
        preds = torch.cat(preds, dim=0)
        tracker.accumulate_metrics(
            gt.permute(0, 2, 3, 1),
            preds.permute(0, 2, 3, 1),
            mask.permute(0, 2, 3, 1),
        )
        if (i + 1) % 100 == 0:
            print(f"  Val: {i+1}/{len(valid_loader)}", flush=True)
    metrics = tracker.get_metrics()
    tracker.reset_metrics()
    return metrics


def _validate_video(model, valid_loader, context, device, sam_mode):
    tracker = RunningMetric(list(DICT_METRICS_DEPTH.keys()))
    for i, batch in enumerate(valid_loader):
        images, depths, masks = _to_clip(batch, device)
        B, T = images.shape[:2]
        preds, valid_masks, valid_depths = [], [], []
        for b in range(B):
            enc_out = model.pixel_encoder(images[b])
            fpn_levels, queries_T = enc_out[:-1], enc_out[-1]
            for t in range(T):
                if masks[b, t].bool().sum() == 0:
                    continue
                frame_fpn = tuple(lvl[t:t + 1] for lvl in fpn_levels)
                inv_frame_fpn = tuple(reversed(frame_fpn))
                with context:
                    pred, _, _ = model(
                        images[b, t:t + 1],
                        instance_queries=queries_T[t],
                        sam_mode=sam_mode,
                        pre_extracted_encoder_outputs=inv_frame_fpn,
                        gt=depths[b, t:t + 1],
                        mask=masks[b, t:t + 1],
                    )
                preds.append(pred)
                valid_masks.append(masks[b, t:t + 1])
                valid_depths.append(depths[b, t:t + 1])
        if preds:
            tracker.accumulate_metrics(
                torch.cat(valid_depths, dim=0).permute(0, 2, 3, 1),
                torch.cat(preds, dim=0).permute(0, 2, 3, 1),
                torch.cat(valid_masks, dim=0).permute(0, 2, 3, 1),
            )
        if (i + 1) % 100 == 0:
            print(f"  Val seq: {i+1}/{len(valid_loader)}", flush=True)
    metrics = tracker.get_metrics()
    tracker.reset_metrics()
    return metrics


def _validate_temporal(model, valid_loader, context, device):
    tracker = RunningMetric(list(DICT_METRICS_DEPTH.keys()))
    flick = []
    for i, batch in enumerate(valid_loader):
        if i >= 40:
            break
        images, depths, masks = _to_clip(batch, device)
        B, T = images.shape[:2]
        for b in range(B):
            clip = images[b]
            labels = model.track_module(clip)
            preds = []
            for t in range(T):
                with context:
                    pred, _, _ = model(clip[t:t + 1], sam_mode=None)
                preds.append(pred)
                if masks[b, t].bool().sum() > 0:
                    tracker.accumulate_metrics(
                        depths[b, t:t + 1].permute(0, 2, 3, 1),
                        pred.permute(0, 2, 3, 1),
                        masks[b, t:t + 1].permute(0, 2, 3, 1),
                    )
            flick.append(temporal_smoothness_loss(torch.cat(preds, dim=0), labels).item())
    metrics = tracker.get_metrics()
    tracker.reset_metrics()
    metrics["flicker"] = float(np.mean(flick)) if flick else 0.0
    return metrics


def _run_validation(model, valid_loader, context, device, sam_mode,
                    dataset_mode, encoder_is_video, temporal_enabled):
    if temporal_enabled:
        return _validate_temporal(model, valid_loader, context, device)
    if dataset_mode == "video" and encoder_is_video:
        return _validate_video(model, valid_loader, context, device, sam_mode)
    return _validate_image(model, valid_loader, context, device, sam_mode)


def _set_trainable(model, encoder_name):
    """Freeze the (frozen) SAM3 backbone but keep iDisc-side modules trainable.
    For the resnet101 baseline this is a no-op — everything trains."""
    for p in model.parameters():
        p.requires_grad = True
    if encoder_name.startswith("sam3"):
        sam_module = getattr(model.pixel_encoder, "sam_model",
                             getattr(model.pixel_encoder, "video_model", None))
        if sam_module is not None:
            for p in sam_module.parameters():
                p.requires_grad = False
        enc = model.pixel_encoder
        for part in getattr(enc, "sam3_trainable", []):
            enc._sam3_submodule(part).requires_grad_(True)
            print(f"  SAM3 unfrozen: {part}", flush=True)

        n_lora = 0
        for m in enc.modules():
            if isinstance(m, LoRALinear):
                m.lora_A.requires_grad_(True)
                m.lora_B.requires_grad_(True)
                n_lora += 1
        if n_lora:
            print(f"  SAM3 trunk LoRA: {n_lora} layers trainable", flush=True)

    if getattr(model, "track_module", None) is not None:
        for p in model.track_module.parameters():
            p.requires_grad = False


def _trainable_state_dict(model, encoder_name):
    """Drop frozen SAM3 weights from checkpoints — they're reloaded from
    sam_checkpoint at startup; persisting them per-save bloats ckpt ~16x."""
    sd = model.state_dict()
    if encoder_name.startswith("sam3"):
        trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
        frozen = ("pixel_encoder.sam_model.", "pixel_encoder.video_model.")
        sd = {k: v for k, v in sd.items()
              if not k.startswith(frozen) or k in trainable_names}
    if getattr(model, "track_module", None) is not None:
        sd = {k: v for k, v in sd.items() if not k.startswith("track_module.vm.")}
    return sd


def _save_ckpt(model, encoder_name, output_dir, name):
    path = os.path.join(output_dir, name)
    torch.save(_trainable_state_dict(model, encoder_name), path)
    return path


def _build_optimizer(model, finetune_cfg, n_iters):
    lr = float(finetune_cfg["lr"])
    sam3_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and ".sam_model." in n]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and ".sam_model." not in n]
    if sam3_params:
        sam3_lr = float(finetune_cfg["sam3_lr"])
        optimizer = optim.AdamW(
            [{"params": head_params, "lr": lr, "weight_decay": 0.01},
             {"params": sam3_params, "lr": sam3_lr, "weight_decay": 0.0}],
        )
        max_lr = [lr, sam3_lr]
        print(f"  LR groups: head {len(head_params)} tensors @ {lr:.1e}, "
              f"SAM3 {len(sam3_params)} tensors @ {sam3_lr:.1e}", flush=True)
    else:
        optimizer = optim.AdamW(head_params, lr=lr, weight_decay=0.01)
        max_lr = lr
    scheduler = OneCycleLR(
        optimizer, max_lr=max_lr, total_steps=n_iters,
        pct_start=0.1, div_factor=10, final_div_factor=100,
    )
    return optimizer, scheduler, head_params + sam3_params


def _train_batch(model, batch, device, context, sam_mode,
                 encoder_is_video, temporal_enabled, temporal_weight):
    if temporal_enabled and "images" in batch:
        images, depths, masks = _to_clip(batch, device)
        vf = _valid_frames(masks)
        if vf == 0:
            return None
        B, T = images.shape[:2]
        total = 0.0
        for b in range(B):
            clip = images[b]
            if temporal_weight > 0:
                labels = model.track_module(clip)
                torch.cuda.empty_cache()
                preds, silog = [], 0.0
                for t in range(T):
                    valid = masks[b, t].bool().sum() > 0
                    with context:
                        pred = checkpoint(lambda f: model(f, sam_mode=None)[0],
                                          clip[t:t + 1], use_reentrant=False)
                        if valid:
                            silog = silog + model.loss.weight * model.loss(
                                pred, target=depths[b, t:t + 1],
                                mask=masks[b, t:t + 1].bool(), interpolate=True)
                    preds.append(pred)
                l_temp = temporal_smoothness_loss(torch.cat(preds, dim=0), labels)
                loss = silog / vf + temporal_weight * l_temp
                loss.backward()
                total += loss.item()
            else:
                for t in range(T):
                    if masks[b, t].bool().sum() == 0:
                        continue
                    with context:
                        _, losses, _ = model(clip[t:t + 1], gt=depths[b, t:t + 1],
                                             mask=masks[b, t:t + 1])
                        loss = sum(v for v in losses["opt"].values())
                    (loss / vf).backward()
                    total += loss.item()
        return total / vf

    if encoder_is_video and "images" in batch:
        images, depths, masks = _to_clip(batch, device)
        vf = _valid_frames(masks)
        if vf == 0:
            return None
        B, T = images.shape[:2]
        total = 0.0
        for b in range(B):
            enc_out = model.pixel_encoder(images[b])
            fpn_levels = enc_out[:-1]
            queries_T = enc_out[-1]
            per_frame = [
                (tuple(lvl[t:t + 1].clone() for lvl in fpn_levels), queries_T[t].clone())
                for t in range(T)
            ]
            del enc_out, fpn_levels, queries_T
            torch.cuda.empty_cache()
            for t in range(T):
                if masks[b, t].bool().sum() == 0:
                    per_frame[t] = None
                    continue
                frame_fpn, iq = per_frame[t]
                per_frame[t] = None
                inv_frame_fpn = tuple(reversed(frame_fpn))
                with context:
                    _, losses, _ = model(
                        images[b, t:t + 1],
                        instance_queries=iq,
                        sam_mode=sam_mode,
                        pre_extracted_encoder_outputs=inv_frame_fpn,
                        gt=depths[b, t:t + 1],
                        mask=masks[b, t:t + 1],
                    )
                    loss = sum(v for v in losses["opt"].values())
                del frame_fpn, iq, inv_frame_fpn
                (loss / vf).backward()
                torch.cuda.empty_cache()
                total += loss.item()
        return total / vf

    data, gt, mask, n_samples = _unpack_batch(batch, device)
    vf = int(mask.reshape(n_samples, -1).bool().any(dim=1).sum().item())
    if vf == 0:
        return None
    total = 0.0
    for idx in range(n_samples):
        # Sequence datasets zero-fill GT for frames without LiDAR; SILog over
        # zero valid pixels is NaN and poisons training.
        if mask[idx].bool().sum() == 0:
            continue
        with context:
            _, losses, _ = model(
                data[idx:idx + 1],
                sam_mode=sam_mode,
                gt=gt[idx:idx + 1],
                mask=mask[idx:idx + 1],
            )
            loss = sum(v for v in losses["opt"].values())
        (loss / vf).backward()
        total += loss.item()
    return total / vf


def run_train(cfg: dict[str, Any]) -> dict[str, Any]:
    sam_mode = cfg["method"]["sam_mode"]
    idr_source = cfg["method"]["idr_source"]
    dataset_mode = cfg["run"]["dataset_mode"]
    encoder_name = cfg["model"]["pixel_encoder"]["name"]
    encoder_is_video = encoder_name == "sam3_video"

    finetune_cfg = cfg["finetune"]
    output_dir = finetune_cfg["output_dir"]
    n_iters = int(finetune_cfg["n_iters"])
    val_interval = int(finetune_cfg["val_interval"])
    batch_size = int(finetune_cfg["batch_size"])
    amp_dtype = _DTYPE_MAP[finetune_cfg["amp_dtype"]]

    temporal_cfg = finetune_cfg.get("temporal") or {}
    temporal_enabled = bool(temporal_cfg.get("enabled", False))
    temporal_weight = float(temporal_cfg.get("weight", 0.0))

    seed = cfg["generic"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Device: {device} | encoder={encoder_name} | sam_mode={sam_mode} | "
          f"dataset_mode={dataset_mode}", flush=True)

    print("Building iDisc model...", flush=True)
    model = IDisc.build(cfg).to(device)
    if idr_source == "afp":
        model.load_pretrained(cfg["paths"]["pretrained_model"])
        print("  iDisc loaded (pretrained weights).", flush=True)
    else:
        print("  iDisc built from scratch (SAM3 backbone frozen).", flush=True)
    if temporal_enabled:
        model.track_module = Sam3TrackModule(
            sam_checkpoint=temporal_cfg["sam_checkpoint"],
            prompt_classes=temporal_cfg["prompt"]["classes"],
            confidence_threshold=temporal_cfg["confidence_threshold"],
        ).to(device)
        print(f"  Temporal tracker side-branch attached (weight={temporal_weight}).", flush=True)
    _set_trainable(model, encoder_name)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} "
          f"({100*trainable/total:.1f}%)", flush=True)

    train_loader = build_train_loader(cfg, batch_size)
    valid_loader = build_val_loader(cfg)
    print(f"  Train: {len(train_loader.dataset)}, Val: {len(valid_loader.dataset)}", flush=True)

    optimizer, scheduler, trainable_params = _build_optimizer(model, finetune_cfg, n_iters)
    amp_enabled = device.type == "cuda" and amp_dtype != torch.float32
    context = torch.autocast(
        device_type=device.type if amp_enabled else "cpu",
        dtype=amp_dtype, enabled=amp_enabled,
    )
    torch.backends.cudnn.benchmark = True

    use_wandb = _WANDB_AVAILABLE and wandb.run is not None
    best_abs_rel = np.inf
    print(f"\nStart training for {n_iters} iters (lr={finetune_cfg['lr']})...", flush=True)
    start = time()
    step = 0
    model.train()
    while step < n_iters:
        for batch in train_loader:
            if step >= n_iters:
                break
            optimizer.zero_grad()
            total_loss = _train_batch(model, batch, device, context, sam_mode,
                                      encoder_is_video, temporal_enabled, temporal_weight)
            if total_loss is None:
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
                if use_wandb:
                    log_metrics(
                        {"global_step": step, "train/loss": total_loss, "train/lr": lr_now},
                        step=step,
                    )

            if step % val_interval == 0 or step == n_iters:
                print(f"\n  Validation at step {step}...", flush=True)
                model.eval()
                with torch.no_grad():
                    metrics = _run_validation(
                        model, valid_loader, context, device, sam_mode,
                        dataset_mode, encoder_is_video, temporal_enabled,
                    )
                model.train()
                _set_trainable(model, encoder_name)

                abs_rel = metrics["abs_rel"]
                print(f"  abs_rel={abs_rel:.6f}  (best={best_abs_rel:.6f})", flush=True)
                for k, v in sorted(metrics.items()):
                    print(f"    {k}: {v:.6f}", flush=True)
                if use_wandb:
                    log_metrics(
                        {"global_step": step, **{f"val/{k}": v for k, v in metrics.items()}},
                        step=step,
                    )

                if abs_rel < best_abs_rel:
                    best_abs_rel = abs_rel
                    ckpt_path = _save_ckpt(model, encoder_name, output_dir, "best_sam_finetuned.pt")
                    print(f"  New best! Saved to {ckpt_path}", flush=True)
                    if use_wandb:
                        log_metrics(
                            {"global_step": step, "val/best_abs_rel": best_abs_rel},
                            step=step,
                        )

                last_path = _save_ckpt(model, encoder_name, output_dir, "last_sam_finetuned.pt")
                print(f"  Checkpoint saved to {last_path}\n", flush=True)

    last_path = _save_ckpt(model, encoder_name, output_dir, "last_sam_finetuned.pt")
    print(f"\nTraining complete. Last model saved to {last_path}", flush=True)
    print(f"Best abs_rel: {best_abs_rel:.6f}", flush=True)

    return {
        "best_abs_rel": float(best_abs_rel),
        "last_checkpoint": last_path,
        "output_dir": output_dir,
        "sam_mode": sam_mode,
        "idr_source": idr_source,
    }
