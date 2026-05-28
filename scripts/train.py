#!/usr/bin/env python
"""Training backend for iDisc (baseline AFP) and SAM3+iDisc variants.

Entered from `run_with_hydra.py` after `build_runtime_config` has validated
the schema. Two execution paths share a single per-frame inner loop:

  - encoder=resnet101 (idr_source=afp)  → standard image dataloader, AFP IDRs.
  - encoder=sam3_image (idr_source=sam3) → SAM3 produces queries per frame;
        replace or translate path consumes them; sequence datasets get flattened.
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
from torch.utils.data import DataLoader, SequentialSampler

import idisc.dataloders as custom_dataset
from idisc.models.idisc import IDisc
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

    # Sequence datasets may expose depth as (depths, masks) instead of (gt, mask).
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
        images = batch["images"].to(device)
        depths = batch["depths"].to(device)
        masks = batch["masks"].to(device)
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


def _trainable_state_dict(model, encoder_name):
    """Drop frozen SAM3 weights from checkpoints — they're reloaded from
    sam_checkpoint at startup; persisting them per-save bloats ckpt ~16x."""
    sd = model.state_dict()
    if encoder_name.startswith("sam3"):
        sd = {k: v for k, v in sd.items()
              if not k.startswith("pixel_encoder.sam_model.")
              and not k.startswith("pixel_encoder.video_model.")}
    return sd


def run_train(cfg: dict[str, Any]) -> dict[str, Any]:
    sam_mode = cfg["method"]["sam_mode"]
    idr_source = cfg["method"]["idr_source"]
    dataset_mode = cfg["run"]["dataset_mode"]
    encoder_name = cfg["model"]["pixel_encoder"]["name"]
    encoder_is_video = encoder_name == "sam3_video"

    finetune_cfg = cfg.get("finetune", {})
    output_dir = finetune_cfg["output_dir"]
    n_iters = int(finetune_cfg["n_iters"])
    lr = float(finetune_cfg["lr"])
    val_interval = int(finetune_cfg["val_interval"])
    batch_size = int(finetune_cfg["batch_size"])
    amp_dtype = _DTYPE_MAP.get(finetune_cfg.get("amp_dtype", "bfloat16"), torch.bfloat16)

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
        # Baseline: load the released resnet101+iDisc weights.
        model.load_pretrained(cfg["paths"]["pretrained_model"])
        print("  iDisc loaded (pretrained weights).", flush=True)
    else:
        print("  iDisc built from scratch (SAM3 backbone frozen).", flush=True)
    _set_trainable(model, encoder_name)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} "
          f"({100*trainable/total:.1f}%)", flush=True)

    dataset_cls = "KITTISequenceDataset" if dataset_mode == "video" else "KITTIDataset"
    data_path = os.path.join(cfg["paths"]["base_path"], cfg["data"]["data_root"])
    print(f"Loading data from {data_path} (dataset_cls={dataset_cls})...", flush=True)
    common_kwargs = dict(
        base_path=data_path,
        crop=cfg["data"].get("crop"),
        manifest_path=cfg["data"].get("manifest_path"),
        clip_length=cfg["data"].get("clip_length", 4),
        stride=cfg["data"].get("stride"),
    )
    train_dataset = getattr(custom_dataset, dataset_cls)(
        test_mode=False,
        augmentations_db=cfg["data"].get("augmentations", {}),
        **common_kwargs,
    )
    valid_dataset = getattr(custom_dataset, dataset_cls)(test_mode=True, **common_kwargs)
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

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    scheduler = OneCycleLR(
        optimizer, max_lr=lr, total_steps=n_iters,
        pct_start=0.1, div_factor=10, final_div_factor=100,
    )
    amp_enabled = device.type == "cuda" and amp_dtype != torch.float32
    context = torch.autocast(
        device_type=device.type if amp_enabled else "cpu",
        dtype=amp_dtype, enabled=amp_enabled,
    )
    torch.backends.cudnn.benchmark = True

    use_wandb = _WANDB_AVAILABLE and wandb.run is not None
    best_abs_rel = np.inf
    print(f"\nStart training for {n_iters} iters (lr={lr})...", flush=True)
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
                images = batch["images"].to(device)
                depths = batch["depths"].to(device)
                masks = batch["masks"].to(device)
                B, T = images.shape[:2]
                valid_frames = int(masks.bool().any(dim=(2, 3, 4)).sum().item())
                if valid_frames == 0:
                    continue
                for b in range(B):
                    enc_out = model.pixel_encoder(images[b])
                    fpn_levels = enc_out[:-1]
                    queries_T = enc_out[-1]
                    per_frame = [
                        (tuple(lvl[t:t + 1].clone() for lvl in fpn_levels),
                         queries_T[t].clone())
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
                        (loss / valid_frames).backward()
                        torch.cuda.empty_cache()
                        total_loss += loss.item()
                total_loss /= valid_frames
            else:
                data, gt, mask, n_samples = _unpack_batch(batch, device)
                valid_frames = int(
                    mask.reshape(n_samples, -1).bool().any(dim=1).sum().item()
                )
                if valid_frames == 0:
                    continue
                for idx in range(n_samples):
                    # Sequence datasets zero-fill GT for frames without LiDAR;
                    # SILog over zero valid pixels is NaN and poisons training.
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
                    (loss / valid_frames).backward()
                    total_loss += loss.item()
                total_loss /= valid_frames

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
                    if dataset_mode == "video" and encoder_is_video:
                        metrics = _validate_video(model, valid_loader, context, device, sam_mode)
                    else:
                        metrics = _validate_image(model, valid_loader, context, device, sam_mode)
                model.train()
                _set_trainable(model, encoder_name)

                abs_rel = metrics.get("abs_rel", np.inf)
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
                    ckpt_path = os.path.join(output_dir, "best_sam_finetuned.pt")
                    torch.save(_trainable_state_dict(model, encoder_name), ckpt_path)
                    print(f"  New best! Saved to {ckpt_path}", flush=True)
                    if use_wandb:
                        log_metrics(
                            {"global_step": step, "val/best_abs_rel": best_abs_rel},
                            step=step,
                        )
                ckpt_path = os.path.join(output_dir, f"checkpoint_step{step}.pt")
                torch.save(_trainable_state_dict(model, encoder_name), ckpt_path)
                print(f"  Checkpoint saved to {ckpt_path}\n", flush=True)

    final_path = os.path.join(output_dir, "final_sam_finetuned.pt")
    torch.save(_trainable_state_dict(model, encoder_name), final_path)
    print(f"\nTraining complete. Final model saved to {final_path}", flush=True)
    print(f"Best abs_rel: {best_abs_rel:.6f}", flush=True)

    return {
        "best_abs_rel": float(best_abs_rel),
        "final_checkpoint": final_path,
        "output_dir": output_dir,
        "sam_mode": sam_mode,
        "idr_source": idr_source,
    }
