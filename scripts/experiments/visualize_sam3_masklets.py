#!/usr/bin/env python
"""
Visualize SAM3 video tracker masklets over KITTI sequences.

Requires Sam3VideoPixelEncoder (is_video_encoder=True) whose forward()
fills encoder._masklets_per_frame — see sam3_video_encoder.py.

Layout per frame (2 rows × 3 cols):
  row 0: [RGB | masklet ID assignment | top-K overlay (frame-0 IDs locked)]
  row 1+: [masklet 0 | masklet 1 | ... ] top-K by score in frame 0
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.cuda as tcuda
import torch.nn.functional as F
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.colors import BoundaryNorm, ListedColormap

sys.path.insert(0, str(Path(__file__).parent))
from visualize_experiments import AttentionCapture, _extract_sample
from visualize_sequence import _isd_assignments, _idr_display

from idisc.dataloders.kitti_sequence import KITTISequenceDataset
from idisc.models.idisc import IDisc

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
TOP_K_MASKLETS = 6   # overridden by --num-masklets at runtime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def denormalize(img_tensor: torch.Tensor) -> np.ndarray:
    mean = IMAGENET_MEAN.to(img_tensor.device)
    std  = IMAGENET_STD.to(img_tensor.device)
    return (img_tensor * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().float().numpy()


def _id_cmap(unique_ids: np.ndarray):
    """Compact colormap keyed on masklet IDs — stable across frames."""
    id_to_idx = {int(i): k for k, i in enumerate(unique_ids.tolist())}
    base   = plt.get_cmap("tab20").colors
    colors = [base[k % 20] for k in range(len(unique_ids))]
    cmap   = ListedColormap(colors)
    norm   = BoundaryNorm(np.arange(len(unique_ids) + 1) - 0.5, len(unique_ids))
    return id_to_idx, cmap, norm


def _assignment_map(masks: torch.Tensor, ids: torch.Tensor, img_hw):
    """Per-pixel masklet ID: argmax over masks, then index into ids."""
    H, W = img_hw
    masks_up = F.interpolate(
        masks.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
    ).squeeze(1)                                   # (M, H, W)
    m_idx  = masks_up.argmax(dim=0)               # (H, W) index into M
    id_map = ids[m_idx.view(-1)].view(H, W).cpu().numpy()
    return id_map, masks_up


def collect_frame(mask_info: dict, img_np: np.ndarray, fixed_ids=None):
    """Build per-frame visualization data.

    fixed_ids: list of int masklet IDs to always show (locked from frame 0).
    """
    masks  = mask_info["masks"]
    ids    = mask_info["ids"]
    scores = mask_info["scores"]

    H, W = img_np.shape[:2]
    if masks is None or ids is None or masks.numel() == 0:
        return None

    _, masks_up = _assignment_map(masks, ids, (H, W))

    if scores is None:
        scores = masks_up.flatten(1).amax(1)

    if fixed_ids is not None:
        id_set = set(ids.tolist())
        top_ids = [i for i in fixed_ids if i in id_set]
        if len(top_ids) < TOP_K_MASKLETS:
            order = torch.argsort(scores, descending=True).tolist()
            for k in order:
                mid = int(ids[k])
                if mid not in top_ids and len(top_ids) < TOP_K_MASKLETS:
                    top_ids.append(mid)
    else:
        order = torch.argsort(scores, descending=True)
        top_ids = [int(ids[k]) for k in order[: min(TOP_K_MASKLETS, ids.shape[0])]]

    # Build top-K masks at full image resolution
    top_masks = []
    top_mask_tensors = []
    for mid in top_ids:
        idx = (ids == mid).nonzero(as_tuple=False)
        if idx.numel():
            m = masks_up[idx[0, 0]]
            top_masks.append(m.cpu().numpy())
            top_mask_tensors.append(m)

    if not top_masks:
        return None

    # Assignment restricted to top_ids only — colors match the overlay exactly.
    # argmax over only the K visible masks; result is an index into top_ids.
    top_stack  = torch.stack(top_mask_tensors, dim=0)        # (K, H, W)
    top_ids_t  = torch.tensor(top_ids, dtype=torch.long)
    best_k     = top_stack.argmax(dim=0)                     # (H, W) index into K
    id_map     = top_ids_t[best_k.view(-1)].view(H, W).numpy()

    return {"id_map": id_map, "top_ids": top_ids, "top_masks": top_masks}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _make_fig():
    n_mask_rows = (TOP_K_MASKLETS + 2) // 3
    n_rows = 1 + 1 + n_mask_rows   # row0=masklet | row1=ISD | row2+=slots
    fig, axes = plt.subplots(n_rows, 3, figsize=(21, 5 * n_rows), constrained_layout=True)
    for ax in axes.flat:
        ax.axis("off")
    return fig, axes


def _init_images(axes, fd, num_heads, num_idrs):
    """Create all AxesImage objects once; return handle dict."""
    img_np = fd["image"]
    fdata  = fd["fdata"]
    all_ids    = np.array(fdata["top_ids"], dtype=np.int64)
    id_to_idx, cmap_id, norm_id = _id_cmap(all_ids)
    colors     = plt.get_cmap("tab20").colors

    # ── row 0: RGB | masklet assignment | top-K overlay ──────────────────────
    axes[0, 0].set_title("RGB", fontsize=12)
    im_rgb = axes[0, 0].imshow(img_np)

    id_idx_map = np.vectorize(lambda v: id_to_idx.get(int(v), 0))(fdata["id_map"])
    axes[0, 1].set_title("Masklet ID assignment", fontsize=12)
    im_asgn = axes[0, 1].imshow(id_idx_map, cmap=cmap_id, norm=norm_id, interpolation="nearest")

    axes[0, 2].set_title(f"Top-{TOP_K_MASKLETS} overlay (frame-0 IDs)", fontsize=12)
    im_ov_base = axes[0, 2].imshow(img_np)
    im_ov_masks = []
    for mask, mid in zip(fdata["top_masks"], fdata["top_ids"]):
        rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
        rgba[..., :3] = colors[id_to_idx[mid] % 20]
        rgba[..., 3]  = mask * 0.5
        im_ov_masks.append(axes[0, 2].imshow(rgba, interpolation="bilinear"))

    # ── row 1: ISD dominant IDR — same code as visualize_sequence.py ─────────
    asgns = _isd_assignments(fd, num_heads, soft=False)
    cmap_idr, norm_idr, vmin_idr, vmax_idr = _idr_display(num_idrs, soft=False)
    im_isd = []
    for col in range(3):
        if col in asgns:
            kwargs = {"cmap": cmap_idr, "interpolation": "nearest"}
            if norm_idr is not None:
                kwargs["norm"] = norm_idr
            else:
                kwargs["vmin"] = vmin_idr
                kwargs["vmax"] = vmax_idr
            im = axes[1, col].imshow(asgns[col], **kwargs)
            axes[1, col].set_title(f"ISD dominant IDR — res {col + 1}", fontsize=12)
            axes[1, col].axis("off")
            im_isd.append(im)
        else:
            im_isd.append(None)

    # ── row 2+: individual masklet panels ─────────────────────────────────────
    im_slot_base, im_slot_mask = [], []
    for i, (mask, mid) in enumerate(zip(fdata["top_masks"], fdata["top_ids"])):
        row, col = 2 + i // 3, i % 3
        axes[row, col].set_title(f"Masklet ID {mid}", fontsize=12)
        im_slot_base.append(axes[row, col].imshow(img_np))
        im_slot_mask.append(axes[row, col].imshow(
            mask, cmap="hot", alpha=0.6, vmin=0, vmax=1, interpolation="bilinear"
        ))

    return {
        "im_rgb": im_rgb, "im_asgn": im_asgn,
        "im_ov_base": im_ov_base, "im_ov_masks": im_ov_masks,
        "im_isd": im_isd,
        "im_slot_base": im_slot_base, "im_slot_mask": im_slot_mask,
        "id_to_idx": id_to_idx, "cmap_id": cmap_id, "norm_id": norm_id,
    }


def _update_images(h, fd, num_heads):
    fdata     = fd["fdata"]
    img_np    = fd["image"]
    id_to_idx = h["id_to_idx"]
    colors    = plt.get_cmap("tab20").colors

    h["im_rgb"].set_data(img_np)
    id_idx_map = np.vectorize(lambda v: id_to_idx.get(int(v), 0))(fdata["id_map"])
    h["im_asgn"].set_data(id_idx_map)
    h["im_ov_base"].set_data(img_np)

    for k, (mask, mid) in enumerate(zip(fdata["top_masks"], fdata["top_ids"])):
        rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
        rgba[..., :3] = colors[id_to_idx.get(mid, 0) % 20]
        rgba[..., 3]  = mask * 0.5
        if k < len(h["im_ov_masks"]):
            h["im_ov_masks"][k].set_data(rgba)
        if k < len(h["im_slot_base"]):
            h["im_slot_base"][k].set_data(img_np)
            h["im_slot_mask"][k].set_data(mask)

    asgns = _isd_assignments(fd, num_heads, soft=False)
    for col, im in enumerate(h["im_isd"]):
        if im is not None and col in asgns:
            im.set_data(asgns[col])


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


def _add_isd_colorbars(fig, axes, h):
    if any(im is not None for im in h["im_isd"]):
        im_ref = next(im for im in h["im_isd"] if im is not None)
        fig.colorbar(im_ref, ax=axes[1, 2], location="right", shrink=0.9, label="IDR index")


def save_frame_png(fd, model_tag, num_heads, num_idrs, path):
    fig, axes = _make_fig()
    h = _init_images(axes, fd, num_heads, num_idrs)
    clip_info = f"clip {fd['clip_idx']}  {fd['seq_id']}"
    fig.colorbar(h["im_asgn"], ax=axes[0, 1], location="right", shrink=0.9,
                 label="masklet index (compressed)")
    _add_isd_colorbars(fig, axes, h)
    fig.suptitle(f"{clip_info}\n{model_tag}", fontsize=11)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def save_animation(frames, model_tag, num_heads, num_idrs, path, fps, fmt):
    fig, axes = _make_fig()
    fd0       = frames[0]
    clip_info = f"clip {fd0['clip_idx']}  {fd0['seq_id']}"
    h = _init_images(axes, fd0, num_heads, num_idrs)
    fig.colorbar(h["im_asgn"], ax=axes[0, 1], location="right", shrink=0.9,
                 label="masklet index (compressed)")
    _add_isd_colorbars(fig, axes, h)
    sup = fig.suptitle(
        f"{clip_info}  — frame 1/{len(frames)}\n{model_tag}", fontsize=11
    )

    def update(t):
        fd = frames[t]
        _update_images(h, fd, num_heads)
        sup.set_text(f"clip {fd['clip_idx']}  {fd['seq_id']}  — frame {t+1}/{len(frames)}\n{model_tag}")
        return []

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 // fps, blit=False
    )
    anim.save(path, writer=_make_writer(fmt, fps))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

def run_clip(model, images, depths, masks, capture, num_heads,
             device, clip_idx, seq_id, sam_mode="replace"):
    enc = model.pixel_encoder
    T   = images.shape[0]
    frames = []

    with torch.no_grad():
        enc.track_masklets = True
        enc_out    = enc(images)           # fills enc._masklets_per_frame
        fpn_levels = enc_out[:-1]
        queries_T  = enc_out[-1]

        fixed_ids = None
        for t in range(T):
            mask_info = enc._masklets_per_frame[t]
            if mask_info is None:
                print(f"  t={t}: no masklet info, skipping")
                continue

            img_np = denormalize(images[t])
            fdata  = collect_frame(mask_info, img_np, fixed_ids=fixed_ids)
            if fdata is None:
                print(f"  t={t}: no valid masklets, skipping")
                continue

            if fixed_ids is None:
                fixed_ids = fdata["top_ids"]

            frame_fpn     = tuple(lvl[t : t + 1] for lvl in fpn_levels)
            inv_frame_fpn = tuple(reversed(frame_fpn))
            capture.reset()
            pred, _, _ = model(
                images[t : t + 1],
                instance_queries=queries_T[t],
                sam_mode=sam_mode,
                pre_extracted_encoder_outputs=inv_frame_fpn,
                gt=depths[t : t + 1], mask=masks[t : t + 1],
            )

            # Store raw ISD attention directly in frame dict — same structure as
            # visualize_sequence.py so _isd_assignments can consume it unchanged.
            isd_attn = {k: v.clone() for k, v in capture.isd_attn.items()}
            isd_hw   = dict(capture.isd_hw)

            frames.append({
                "image":    img_np,
                "clip_idx": clip_idx,
                "seq_id":   seq_id,
                "fdata":    fdata,
                "isd_attn": isd_attn,
                "isd_hw":   isd_hw,
            })

    return frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(cfg):
    with open(cfg["config_file"]) as f:
        config = json.load(f)

    output_dir  = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device      = torch.device("cuda") if tcuda.is_available() else torch.device("cpu")
    num_clips   = cfg.get("num_clips", 3)
    start_clip  = cfg.get("start_clip", 0)
    fps         = cfg.get("fps", 3)
    fmt         = cfg.get("format", "gif")
    clip_length = cfg.get("clip_length") or config["data"].get("clip_length", 4)
    sam_mode    = cfg.get("sam_mode", "replace")
    model_tag   = f"{Path(cfg['config_file']).stem}  masklets  sam_mode={sam_mode}"

    global TOP_K_MASKLETS
    TOP_K_MASKLETS = cfg.get("num_masklets", TOP_K_MASKLETS)

    num_heads = config["model"]["num_heads"]

    model = IDisc.build(config)
    model.load_pretrained(cfg["model_file"])
    model = model.to(device).eval()

    if not getattr(model.pixel_encoder, "is_video_encoder", False):
        raise RuntimeError(
            "This script requires Sam3VideoPixelEncoder (is_video_encoder=True). "
            "Use a config with pixel_encoder.name=sam3_video."
        )

    capture = AttentionCapture(model)

    manifest  = cfg.get("manifest_path") or config["data"].get(
        "manifest_path", "splits/kitti/sequence_manifest.json"
    )
    data_path = os.path.join(cfg["base_path"], config["data"]["data_root"])
    dataset   = KITTISequenceDataset(
        test_mode=True,
        base_path=data_path,
        manifest_path=os.path.join(cfg["base_path"], manifest),
        clip_length=clip_length,
        crop=config["data"]["crop"],
    )
    print(f"{len(dataset)} clips — start={start_clip}  n={num_clips}")

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
        frames = run_clip(model, images, depths, masks, capture, num_heads,
                          device, clip_idx, seq_id, sam_mode)
        print(f"  {len(frames)} frames")
        if not frames:
            continue

        clip_dir = output_dir / f"clip_{clip_idx:03d}_{seq_id}_{sam_mode}"
        clip_dir.mkdir(exist_ok=True)
        # num_idrs from first frame that has ISD data
        num_idrs = config["model"]["afp"]["num_latents"]
        for fd in frames:
            if fd.get("isd_attn"):
                first_res = min(fd["isd_attn"].keys())
                num_idrs = _extract_sample(
                    fd["isd_attn"][first_res], num_heads, 0
                ).shape[-1]
                break

        for t, fd in enumerate(frames):
            path = clip_dir / f"frame_{t:03d}.png"
            save_frame_png(fd, model_tag, num_heads, num_idrs, path)
            print(f"  saved {path.name}")

        anim_path = output_dir / f"clip_{clip_idx:03d}_{seq_id}_{sam_mode}.{fmt}"
        save_animation(frames, model_tag, num_heads, num_idrs, anim_path, fps, fmt)
        print(f"  saved {anim_path.name}")

    capture.remove()
    print(f"\nDone → {output_dir}")


def _parse_args():
    p = argparse.ArgumentParser(description="Visualize SAM3 video tracker masklets")
    p.add_argument("--config-file",   required=True)
    p.add_argument("--model-file",    required=True)
    p.add_argument("--base-path",     required=True)
    p.add_argument("--output-dir",    default="viz_masklets")
    p.add_argument("--manifest-path", default=None)
    p.add_argument("--num-clips",     type=int, default=3)
    p.add_argument("--start-clip",    type=int, default=0)
    p.add_argument("--clip-length",   type=int, default=None)
    p.add_argument("--num-masklets",  type=int, default=6,
                   help="How many top-scoring masklets to show per frame")
    p.add_argument("--sam-mode",      default="replace",
                   choices=["replace", "concat", "translate", "none"],
                   help="IDR source for depth head. 'replace' is safe for the video encoder "
                        "as it bypasses AFP (which has fewer resolutions than ISD).")
    p.add_argument("--fps",           type=int, default=3)
    p.add_argument("--format",        default="gif", choices=["gif", "mp4"])
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
        "num_masklets":  args.num_masklets,
        "sam_mode":      args.sam_mode,
        "fps":           args.fps,
        "format":        args.format,
    })


if __name__ == "__main__":
    main()
