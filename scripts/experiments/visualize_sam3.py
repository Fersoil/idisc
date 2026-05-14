#!/usr/bin/env python
"""
Visualize SAM3 mask predictions and token queries for each frame in a sequence.

Uses two capture points — both require no changes to any model code:
  1. model.pixel_encoder.sam_model  forward hook  → pred_masks (B, Q, H, W)
     and pred_logits (B, Q, C), available from the raw model output dict.
  2. model.pixel_encoder._decoder_hs_buf          → query embeddings (Q, 256),
     already filled by the decoder hook wired in Sam3PixelEncoder.

Layout per frame (2 rows × 3 cols):
  row 0: [RGB | SAM3 slot assignment | top-score mask overlay]
  row 1: [query 0 mask | query 1 mask | query 2 mask]   (highest-score slots)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
import torch.cuda as tcuda
import torch.nn.functional as F
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.colors import BoundaryNorm, ListedColormap
from torch.utils.data import DataLoader, SequentialSampler

from idisc.dataloders.kitti_sequence import KITTISequenceDataset
from idisc.models.idisc import IDisc

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

TOP_K_MASKS = 6   # individual mask panels in rows 1-2


# ---------------------------------------------------------------------------
# SAM3 output capture via forward hook on sam_model
# ---------------------------------------------------------------------------

class Sam3Capture:
    """
    Patches sam_model.forward_grounding to capture pred_masks and pred_logits.

    register_forward_hook on sam_model doesn't fire because the processor calls
    forward_grounding() directly (not __call__).  Patching the method on the
    instance is the reliable approach — forward_grounding always returns (out, hs)
    where out contains pred_masks and pred_logits.

    Query embeddings are read from encoder._decoder_hs_buf which is already
    populated by Sam3PixelEncoder._register_decoder_hook().
    """

    def __init__(self, encoder):
        self.encoder  = encoder
        # Queue: one entry per forward_grounding call.
        # Image encoder: one call per model() forward (reset before each frame).
        # Video encoder: T calls during pixel_encoder(clip) — pop per frame.
        self._queue: list = []

        sam_model = encoder.sam_model
        orig_fg   = sam_model.forward_grounding
        capture   = self

        def _patched_fg(*args, **kwargs):
            out, hs = orig_fg(*args, **kwargs)
            entry = {}
            if "pred_masks" in out:
                entry["masks"]  = out["pred_masks"][0].detach().cpu().float()
            if "pred_logits" in out:
                entry["logits"] = out["pred_logits"][0].detach().cpu().float()
            capture._queue.append(entry)
            return out, hs

        sam_model.forward_grounding = _patched_fg
        self._sam_model = sam_model
        self._orig_fg   = orig_fg

    def reset(self):
        self._queue.clear()

    def pop(self):
        """Return and remove the oldest captured frame entry."""
        return self._queue.pop(0) if self._queue else {}

    @property
    def query_embeddings(self):
        hs = self.encoder._decoder_hs_buf
        if hs is None:
            return None
        return hs.squeeze(1).float().cpu() if hs.dim() == 3 else hs.float().cpu()

    def remove(self):
        self._sam_model.forward_grounding = self._orig_fg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def denormalize(img_tensor):
    mean = IMAGENET_MEAN.to(img_tensor.device)
    std  = IMAGENET_STD.to(img_tensor.device)
    return (img_tensor * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().float().numpy()


def _slot_cmap(n):
    base   = plt.get_cmap("tab20").colors
    colors = [base[i % 20] for i in range(n)]
    cmap   = ListedColormap(colors)
    norm   = BoundaryNorm(np.arange(n + 1) - 0.5, n)
    return cmap, norm


def _collect_frame(entry, img_hw, sam_mode="translate", fixed_slots=None):
    """Return dict of arrays for one frame, all at full image resolution.

    fixed_slots: if provided, always show these slot indices (locked from frame 0).
    """
    H, W  = img_hw
    masks  = entry.get("masks")    # (Q, H_low, W_low)
    logits = entry.get("logits")   # (Q, C)

    if masks is None:
        return None

    Q = masks.shape[0]
    masks_up = F.interpolate(
        masks.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
    ).squeeze(0)   # (Q, H, W)

    # SAM3 may return probabilities [0,1] or raw logits (unbounded).
    # Apply sigmoid only when values are clearly logits (outside [0,1]).
    if masks_up.min() < -0.1 or masks_up.max() > 1.1:
        probs = masks_up.sigmoid()
    else:
        probs = masks_up.clamp(0, 1)

    assignment = probs.argmax(dim=0).numpy()   # (H, W) int

    # Top-K by class score; fall back to mask magnitude if no logits
    if logits is not None:
        scores = logits.sigmoid().max(dim=-1).values   # (Q,)
    else:
        scores = probs.amax(dim=(1, 2))                # (Q,) — peak activation

    if fixed_slots is not None:
        # Keep the same slots as frame 0 to avoid inter-frame flickering.
        topk_idx = [i for i in fixed_slots if i < Q]
        # Pad with highest-scoring remaining slots if needed.
        if len(topk_idx) < TOP_K_MASKS:
            ranked = scores.argsort(descending=True).tolist()
            for i in ranked:
                if i not in topk_idx and len(topk_idx) < TOP_K_MASKS:
                    topk_idx.append(i)
    else:
        topk_idx = scores.topk(min(TOP_K_MASKS, Q)).indices.tolist()

    topk_masks = [probs[i].numpy() for i in topk_idx]

    return {
        "assignment": assignment,
        "topk_masks": topk_masks,
        "topk_idx":   topk_idx,
        "num_slots":  Q,
        "isd_label":  _ISD_LABELS.get(sam_mode, "IDR assignment"),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _make_fig():
    n_mask_rows = (TOP_K_MASKS + 2) // 3   # ceil(TOP_K_MASKS / 3)
    n_rows = 1 + n_mask_rows
    fig, axes = plt.subplots(n_rows, 3, figsize=(21, 4 * n_rows), constrained_layout=True)
    for ax in axes.flat:
        ax.axis("off")
    return fig, axes


def _init_images(axes, img_np, sam_data):
    """Create every AxesImage object once with fixed norm/cmap. Returns handles dict."""
    Q          = sam_data["num_slots"]
    cmap, norm = _slot_cmap(Q)
    colors     = plt.get_cmap("tab20").colors

    axes[0, 0].set_title("RGB", fontsize=12)
    im_rgb = axes[0, 0].imshow(img_np)

    axes[0, 1].set_title(sam_data.get("isd_label", "SAM3 slot assignment"), fontsize=12)
    im_asgn = axes[0, 1].imshow(
        sam_data["assignment"], cmap=cmap, norm=norm, interpolation="nearest"
    )

    axes[0, 2].set_title(f"Top-{TOP_K_MASKS} masks overlay (frame-0 slots)", fontsize=12)
    im_ov_rgb = axes[0, 2].imshow(img_np)
    im_ov_masks = []
    for mask, slot in zip(sam_data["topk_masks"], sam_data["topk_idx"]):
        rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
        rgba[..., :3] = colors[slot % 20]
        rgba[..., 3]  = mask * 0.5
        im_ov_masks.append(axes[0, 2].imshow(rgba, interpolation="bilinear"))

    im_slot_rgb, im_slot_mask = [], []
    for i, (mask, slot) in enumerate(zip(sam_data["topk_masks"], sam_data["topk_idx"])):
        row, col = 1 + i // 3, i % 3
        axes[row, col].set_title(f"Slot {slot} mask", fontsize=12)
        im_slot_rgb.append(axes[row, col].imshow(img_np))
        im_slot_mask.append(axes[row, col].imshow(
            mask, cmap="hot", alpha=0.6, vmin=0, vmax=1, interpolation="bilinear"
        ))

    return {
        "rgb":       im_rgb,
        "asgn":      im_asgn,
        "ov_rgb":    im_ov_rgb,
        "ov_masks":  im_ov_masks,
        "slot_rgb":  im_slot_rgb,
        "slot_mask": im_slot_mask,
    }


def _update_images(handles, img_np, sam_data):
    """Update all panels in-place via set_data — no new AxesImage objects created."""
    colors = plt.get_cmap("tab20").colors

    handles["rgb"].set_data(img_np)
    handles["asgn"].set_data(sam_data["assignment"])
    handles["ov_rgb"].set_data(img_np)

    for k, (mask, slot) in enumerate(zip(sam_data["topk_masks"], sam_data["topk_idx"])):
        rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
        rgba[..., :3] = colors[slot % 20]
        rgba[..., 3]  = mask * 0.5
        handles["ov_masks"][k].set_data(rgba)
        handles["slot_rgb"][k].set_data(img_np)
        handles["slot_mask"][k].set_data(mask)


def _save_frame_png(img_np, fd, sam_data, model_tag, path):
    fig, axes = _make_fig()
    handles   = _init_images(axes, img_np, sam_data)
    fig.colorbar(handles["asgn"], ax=axes[0, 1], location="right", shrink=0.9, label="slot index")
    clip_line = f"clip {fd.get('clip_idx', '')}  {fd.get('seq_id', '')}"
    fig.suptitle(f"{clip_line}\n{model_tag}", fontsize=11)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _save_animation(frames, model_tag, path, fps, fmt):
    fig, axes = _make_fig()
    fd0       = frames[0]
    handles   = _init_images(axes, fd0["image"], fd0["sam_data"])
    fig.colorbar(handles["asgn"], ax=axes[0, 1], location="right", shrink=0.9, label="slot index")
    sup = fig.suptitle(
        f"clip {fd0['clip_idx']}  {fd0['seq_id']}  — frame 1/{len(frames)}\n{model_tag}",
        fontsize=11,
    )

    def update(t):
        fd = frames[t]
        _update_images(handles, fd["image"], fd["sam_data"])
        sup.set_text(
            f"clip {fd['clip_idx']}  {fd['seq_id']}  — frame {t + 1}/{len(frames)}\n{model_tag}"
        )
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 // fps, blit=False)
    writer = _make_writer(fmt, fps)
    anim.save(path, writer=writer)
    plt.close(fig)


def _make_writer(fmt, fps):
    if fmt == "mp4":
        import shutil
        if shutil.which("ffmpeg") is None:
            try:
                import imageio_ffmpeg
                plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                print("Warning: ffmpeg not found, falling back to GIF")
                return PillowWriter(fps=fps)
        return FFMpegWriter(fps=fps)
    return PillowWriter(fps=fps)


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

_ISD_LABELS = {
    "replace":   "SAM3 IDR assignment",
    "translate": "SAM3→IDR assignment",
    "concat":    "AFP+SAM3 IDR assignment",
    "none":      "AFP IDR assignment",
}


def _run_clip(model, capture, images, depths, masks, device, clip_idx, seq_id, sam_mode="translate"):
    T = images.shape[0]
    encoder_is_video = getattr(model.pixel_encoder, "is_video_encoder", False)
    frames = []

    with torch.no_grad():
        if encoder_is_video:
            # forward_grounding is called T times inside pixel_encoder(images).
            # Collect all T entries into the queue, then pop per frame below.
            capture.reset()
            enc_out    = model.pixel_encoder(images)
            fpn_levels = enc_out[:-1]
            queries_T  = enc_out[-1]
            print(f"  encoder captured {len(capture._queue)} mask entries")

        fixed_slots = None   # locked from first valid frame; stable across all frames
        for t in range(T):
            frame = images[t : t + 1]
            gt    = depths[t : t + 1]
            msk   = masks[t : t + 1]

            if encoder_is_video:
                frame_fpn     = tuple(lvl[t : t + 1] for lvl in fpn_levels)
                inv_frame_fpn = tuple(reversed(frame_fpn))
                entry = capture.pop()   # masks were captured during encoder forward
                pred, _, _ = model(
                    frame,
                    instance_queries=queries_T[t],
                    sam_mode=sam_mode,
                    pre_extracted_encoder_outputs=inv_frame_fpn,
                    gt=gt, mask=msk,
                )
            else:
                capture.reset()         # clear before each frame
                pred, _, _ = model(frame, sam_mode=sam_mode, gt=gt, mask=msk)
                entry = capture.pop()   # single entry added during this forward

            img_np   = denormalize(frame[0])
            H, W     = img_np.shape[:2]
            sam_data = _collect_frame(entry, (H, W), sam_mode, fixed_slots=fixed_slots)
            if sam_data is None:
                print(f"  frame {t}: no SAM3 masks captured, skipping")
                continue

            if fixed_slots is None:
                fixed_slots = sam_data["topk_idx"]   # lock slot selection to frame 0

            frames.append({
                "image":    img_np,
                "clip_idx": clip_idx,
                "seq_id":   seq_id,
                "sam_data": sam_data,
            })

    return frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(cfg):
    with open(cfg["config_file"]) as f:
        config = json.load(f)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device     = torch.device("cuda") if tcuda.is_available() else torch.device("cpu")
    num_clips  = cfg.get("num_clips", 3)
    start_clip = cfg.get("start_clip", 0)
    fps        = cfg.get("fps", 3)
    fmt        = cfg.get("format", "gif")
    clip_length = cfg.get("clip_length") or config["data"].get("clip_length", 4)
    sam_mode   = cfg.get("sam_mode", "translate")
    model_tag  = f"{Path(cfg['config_file']).stem}  sam_mode={sam_mode}"
    isd_label  = _ISD_LABELS.get(sam_mode, "IDR assignment")

    model = IDisc.build(config)
    model.load_pretrained(cfg["model_file"])
    model = model.to(device).eval()

    if not getattr(model.pixel_encoder, "yields_instance_queries", False):
        raise RuntimeError("This script requires a SAM3-based encoder "
                           "(yields_instance_queries=True). Use kitti_sam3_translate.json.")

    capture = Sam3Capture(model.pixel_encoder)

    manifest = cfg.get("manifest_path") or config["data"].get(
        "manifest_path", "splits/kitti/sequence_manifest.json"
    )
    data_path = os.path.join(cfg["base_path"], config["data"]["data_root"])
    dataset = KITTISequenceDataset(
        test_mode=True,
        base_path=data_path,
        manifest_path=os.path.join(cfg["base_path"], manifest),
        clip_length=clip_length,
        crop=config["data"]["crop"],
    )
    print(f"{len(dataset)} clips  —  start={start_clip}  n={num_clips}")

    if start_clip >= len(dataset):
        raise ValueError(f"--start-clip {start_clip} >= dataset size {len(dataset)}")
    end_clip = min(start_clip + num_clips, len(dataset))
    if end_clip < start_clip + num_clips:
        print(f"Warning: only {end_clip - start_clip} clips available from {start_clip}")

    for clip_idx in range(start_clip, end_clip):
        clip   = dataset[clip_idx]
        images = clip["images"].to(device)
        depths = clip["depths"].to(device)
        masks  = clip["masks"].to(device)
        seq_id = clip["sequence_id"].replace("/", "_")

        print(f"\nClip {clip_idx}  seq={seq_id}")
        frames = _run_clip(model, capture, images, depths, masks, device, clip_idx, seq_id, sam_mode)
        print(f"  {len(frames)} frames collected")
        if not frames:
            continue

        clip_dir = output_dir / f"clip_{clip_idx:03d}_{seq_id}"
        clip_dir.mkdir(exist_ok=True)
        for t, fd in enumerate(frames):
            path = clip_dir / f"frame_{t:03d}.png"
            _save_frame_png(fd["image"], fd, fd["sam_data"], model_tag, path)
            print(f"  saved {path.name}")

        anim_path = output_dir / f"clip_{clip_idx:03d}_{seq_id}_{sam_mode}.{fmt}"
        _save_animation(frames, model_tag, anim_path, fps, fmt)
        print(f"  saved {anim_path.name}")

    capture.remove()
    print(f"\nDone → {output_dir}")


def _parse_args():
    p = argparse.ArgumentParser(description="Visualize SAM3 masks and token queries")
    p.add_argument("--config-file",    required=True)
    p.add_argument("--model-file",     required=True)
    p.add_argument("--base-path",      required=True)
    p.add_argument("--output-dir",     default="viz_sam3")
    p.add_argument("--manifest-path",  default=None)
    p.add_argument("--num-clips",      type=int, default=3)
    p.add_argument("--start-clip",     type=int, default=0)
    p.add_argument("--clip-length",    type=int, default=None)
    p.add_argument("--sam-mode",       default="translate",
                   choices=["translate", "replace", "concat", "none"])
    p.add_argument("--fps",            type=int, default=3)
    p.add_argument("--format",         default="gif", choices=["gif", "mp4"])
    return p.parse_args()


def main():
    args = _parse_args()
    run({
        "config_file":   args.config_file,
        "model_file":    args.model_file,
        "base_path":     args.base_path,
        "output_dir":    args.output_dir,
        "manifest_path": args.manifest_path,
        "num_clips":     args.num_clips,
        "start_clip":    args.start_clip,
        "clip_length":   args.clip_length,
        "sam_mode":      args.sam_mode,
        "fps":           args.fps,
        "format":        args.format,
    })


if __name__ == "__main__":
    main()
