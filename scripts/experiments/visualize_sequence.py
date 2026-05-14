#!/usr/bin/env python
"""
Visualize IDR attention maps over a KITTI sequence clip.

For each clip: runs depth estimation frame-by-frame (handling both image and
video encoders), captures AFP and ISD attention via forward hooks, saves
per-frame PNGs and a 4-panel GIF animation:
  [RGB | depth | ISD dominant IDR | AFP slot heatmap]
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
from visualize_experiments import (
    AttentionCapture,
    _extract_sample,
    _to_spatial,
    denormalize_imagenet,
    _save_idr_assignment,
)

from idisc.dataloders.kitti_sequence import KITTISequenceDataset
from idisc.models.idisc import IDisc


# ---------------------------------------------------------------------------
# Per-frame inference
# ---------------------------------------------------------------------------

def _run_clip(model, images, depths, masks, capture, device, clip_idx=0, seq_id=""):
    """
    Run inference on one clip (T, 3, H, W).  Handles both video encoders
    (full-clip backbone call, then per-frame dispatch) and image encoders
    (straightforward per-frame call).

    Returns a list of dicts — one per frame — with keys:
      image, depth, isd_attn, isd_hw, afp_attn, afp_hw
    """
    T = images.shape[0]
    encoder_is_video = getattr(model.pixel_encoder, "is_video_encoder", False)

    frames = []

    with torch.no_grad():
        if encoder_is_video:
            enc_out = model.pixel_encoder(images)
            fpn_levels = enc_out[:-1]   # each (T, C, h, w)
            queries_T  = enc_out[-1]    # (T, K, 256)

        for t in range(T):
            capture.reset()
            frame = images[t : t + 1]
            gt    = depths[t : t + 1]
            msk   = masks[t : t + 1]

            if encoder_is_video:
                frame_fpn     = tuple(lvl[t : t + 1] for lvl in fpn_levels)
                inv_frame_fpn = tuple(reversed(frame_fpn))
                iq            = queries_T[t]
                pred, _, _ = model(
                    frame,
                    instance_queries=iq,
                    pre_extracted_encoder_outputs=inv_frame_fpn,
                    gt=gt, mask=msk,
                )
            else:
                pred, _, _ = model(frame, gt=gt, mask=msk)

            frames.append({
                "image":    denormalize_imagenet(frame[0]),
                "depth":    pred[0, 0].cpu().float().numpy(),
                "gt":       gt[0, 0].cpu().float().numpy(),
                "clip_idx": clip_idx,
                "seq_id":   seq_id,
                "isd_attn": {k: v.clone() for k, v in capture.isd_attn.items()},
                "isd_hw":   dict(capture.isd_hw),
                "afp_attn": {k: v.clone() for k, v in capture.afp_attn.items()},
                "afp_hw":   dict(capture.afp_hw),
            })

    return frames


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _isd_assignments(fd, num_heads):
    """Return {res_idx: (H_feat, W_feat) int array} for all ISD resolutions."""
    out = {}
    for res_idx, attn_bh in fd["isd_attn"].items():
        h, w = fd["isd_hw"][res_idx]
        attn = _extract_sample(attn_bh, num_heads, sample_idx=0)  # (N_pixels, N_IDR)
        out[res_idx] = attn.argmax(dim=-1).reshape(h, w).numpy()
    return out


def _make_fig(frames_data, num_heads, depth_max, num_idrs):
    """Build the figure/axes layout: [RGB | Depth | ISD res 1 | res 2 | ...]."""
    n_isd_res = len(frames_data[0]["isd_attn"])
    n_cols    = 2 + n_isd_res
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5))
    axes[0].set_title("RGB",   fontsize=11)
    axes[1].set_title("Depth", fontsize=11)
    for r in range(n_isd_res):
        axes[2 + r].set_title(f"ISD dominant IDR — res {r + 1}", fontsize=11)
    for ax in axes:
        ax.axis("off")
    return fig, axes, n_isd_res


def _make_axes():
    """3 columns × 2 rows. Returns fig and flat axes list (row-major)."""
    fig, axes = plt.subplots(2, 3, figsize=(21, 10), constrained_layout=True)
    for ax in axes.flat:
        ax.axis("off")
    return fig, axes


def _depth_cmap():
    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad(color="#444444")   # dark grey for missing GT pixels
    return cmap


def _idr_cmap_norm(num_idrs):
    """Discrete colormap with one fixed colour per IDR index — consistent across frames.

    Cycles through the 20 tab20 colours explicitly so indices above 20 never
    get interpolated intermediate colours (which is what get_cmap("tab20", N>20) does).
    """
    base = plt.get_cmap("tab20").colors          # exactly 20 RGBA tuples, no interpolation
    colors = [base[i % 20] for i in range(num_idrs)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(num_idrs + 1) - 0.5, num_idrs)
    return cmap, norm


def _fill_axes(axes, fd, num_heads, depth_max, num_idrs, cmap_depth, cmap_idr, norm_idr):
    """
    Layout:
      row 0: [RGB | depth pred | GT depth]   — shared plasma scale; zeros masked grey
      row 1: [ISD res 0 | ISD res 1 | ISD res 2]  — discrete fixed colormap
    """
    asgns = _isd_assignments(fd, num_heads)

    # row 0
    axes[0, 0].imshow(fd["image"])
    axes[0, 0].set_title("RGB", fontsize=12)

    im_d = axes[0, 1].imshow(fd["depth"], cmap=cmap_depth, vmin=0, vmax=depth_max)
    axes[0, 1].set_title("Depth prediction", fontsize=12)

    gt_masked = np.ma.masked_equal(fd["gt"], 0)
    im_gt = axes[0, 2].imshow(gt_masked, cmap=cmap_depth, vmin=0, vmax=depth_max)
    axes[0, 2].set_title("GT depth  (grey = no annotation)", fontsize=12)

    # row 1 — use imshow + fixed discrete norm so colours never flicker
    im_asgns = []
    for col in range(3):
        if col in asgns:
            im = axes[1, col].imshow(
                asgns[col], cmap=cmap_idr, norm=norm_idr, interpolation="nearest"
            )
            axes[1, col].set_title(f"ISD dominant IDR — res {col + 1}", fontsize=12)
            axes[1, col].axis("off")
            im_asgns.append(im)

    return im_d, im_gt, im_asgns


def _save_frame_png(fd, num_heads, depth_max, num_idrs, path):
    cmap_depth = _depth_cmap()
    cmap_idr, norm_idr = _idr_cmap_norm(num_idrs)
    fig, axes = _make_axes()
    im_d, im_gt, im_asgns = _fill_axes(
        axes, fd, num_heads, depth_max, num_idrs, cmap_depth, cmap_idr, norm_idr
    )
    fig.colorbar(im_d, ax=axes[0, 2], location="right", shrink=0.9, label="depth (m)")
    if im_asgns:
        cb = fig.colorbar(im_asgns[-1], ax=axes[1, 2], location="right", shrink=0.9)
        cb.set_label("IDR index")
        step = max(1, num_idrs // 10)
        cb.set_ticks(range(0, num_idrs, step))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _make_writer(fmt: str, fps: int):
    if fmt == "mp4":
        return FFMpegWriter(fps=fps, metadata={"title": "IDR sequence"})
    return PillowWriter(fps=fps)


def _save_gif(frames_data, num_heads, depth_max, num_idrs, path, fps, fmt="gif"):
    cmap_depth = _depth_cmap()
    cmap_idr, norm_idr = _idr_cmap_norm(num_idrs)
    fig, axes = _make_axes()
    im_d, im_gt, im_asgns = _fill_axes(
        axes, frames_data[0], num_heads, depth_max, num_idrs, cmap_depth, cmap_idr, norm_idr
    )
    fig.colorbar(im_d, ax=axes[0, 2], location="right", shrink=0.9, label="depth (m)")
    if im_asgns:
        cb = fig.colorbar(im_asgns[-1], ax=axes[1, 2], location="right", shrink=0.9)
        cb.set_label("IDR index")
        step = max(1, num_idrs // 10)
        cb.set_ticks(range(0, num_idrs, step))
    seq_id = frames_data[0].get("seq_id", "")
    sup = fig.suptitle(f"clip {frames_data[0].get('clip_idx', '')}  {seq_id}  — frame 1", fontsize=13)

    def update(t):
        fd    = frames_data[t]
        asgns = _isd_assignments(fd, num_heads)
        axes[0, 0].images[0].set_data(fd["image"])
        im_d.set_data(fd["depth"])
        im_gt.set_data(np.ma.masked_equal(fd["gt"], 0))
        for col, im in enumerate(im_asgns):
            im.set_data(asgns[col])
        fd = frames_data[t]
        sup.set_text(f"clip {fd.get('clip_idx', '')}  {fd.get('seq_id', '')}  — frame {t + 1}/{len(frames_data)}")
        return [im_d, im_gt, *im_asgns, sup]

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames_data), interval=1000 // fps, blit=False
    )
    anim.save(path, writer=_make_writer(fmt, fps))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_sequence_visualization(cfg: dict) -> None:
    with open(cfg["config_file"], "r", encoding="utf-8") as f:
        config = json.load(f)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device     = torch.device("cuda") if tcuda.is_available() else torch.device("cpu")
    num_heads   = config["model"]["num_heads"]
    num_clips   = cfg.get("num_clips", 3)
    start_clip  = cfg.get("start_clip", 0)
    fps         = cfg.get("fps", 2)
    fmt         = cfg.get("format", "gif")

    model = IDisc.build(config)
    model.load_pretrained(cfg["model_file"])
    model = model.to(device).eval()
    print(f"Model loaded — {num_heads} heads, {model.afp.num_resolutions} AFP resolutions")

    clip_length = cfg["clip_length"] if cfg.get("clip_length") is not None else config["data"].get("clip_length", 4)
    data_path = os.path.join(cfg["base_path"], config["data"]["data_root"])
    dataset = KITTISequenceDataset(
        test_mode=True,
        base_path=data_path,
        manifest_path=os.path.join(cfg["base_path"], config["data"]["manifest_path"]),
        clip_length=clip_length,
        crop=config["data"]["crop"],
    )
    print(f"{len(dataset)} clips — visualizing first {num_clips}")

    capture = AttentionCapture(model)

    if start_clip >= len(dataset):
        raise ValueError(
            f"--start-clip {start_clip} is out of range: dataset only has {len(dataset)} clips"
        )
    end_clip = min(start_clip + num_clips, len(dataset))
    if end_clip < start_clip + num_clips:
        print(f"Warning: requested {num_clips} clips but only {end_clip - start_clip} available "
              f"from clip {start_clip} (dataset has {len(dataset)} clips)")
    print(f"Visualizing clips {start_clip}–{end_clip - 1} of {len(dataset)}")
    for clip_idx in range(start_clip, end_clip):
        clip       = dataset[clip_idx]
        images     = clip["images"].to(device)
        depths     = clip["depths"].to(device)
        masks      = clip["masks"].to(device)
        seq_id     = clip["sequence_id"].replace("/", "_")
        T          = images.shape[0]

        print(f"\nClip {clip_idx}  seq={seq_id}  frames={clip['frame_indices']}")
        frames_data = _run_clip(model, images, depths, masks, capture, device, clip_idx, seq_id)
        print(f"  {T} frames collected")

        # Compute global ranges once so colormaps are consistent across frames
        depth_max    = max(fd["depth"].max() for fd in frames_data)
        first_res    = min(frames_data[0]["isd_attn"].keys())
        num_idrs     = _extract_sample(
            frames_data[0]["isd_attn"][first_res], num_heads, 0
        ).shape[-1]

        clip_dir = output_dir / f"clip_{clip_idx:03d}_{seq_id}"
        clip_dir.mkdir(exist_ok=True)

        for t, fd in enumerate(frames_data):
            png_path = clip_dir / f"frame_{t:03d}.png"
            _save_frame_png(fd, num_heads, depth_max, num_idrs, png_path)
            print(f"  saved {png_path.name}")

        anim_path = output_dir / f"clip_{clip_idx:03d}_{seq_id}.{fmt}"
        _save_gif(frames_data, num_heads, depth_max, num_idrs, anim_path, fps, fmt)
        print(f"  saved {anim_path.name}")

    capture.remove()
    print(f"\nDone → {output_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize IDR attention over KITTI sequences")
    parser.add_argument("--config-file",  required=True)
    parser.add_argument("--model-file",   required=True)
    parser.add_argument("--base-path",    required=True)
    parser.add_argument("--output-dir",   default="viz_sequence")
    parser.add_argument("--num-clips",   type=int, default=3)
    parser.add_argument("--start-clip",  type=int, default=0,
                        help="Index of the first clip in the dataset to visualize")
    parser.add_argument("--clip-length", type=int, default=None,
                        help="Override clip_length in the config JSON and save it back")
    parser.add_argument("--fps",    type=int, default=2)
    parser.add_argument("--format", default="gif", choices=["gif", "mp4"],
                        help="Output format — gif has centisecond delay resolution "
                             "so fps above ~10 won't be exact; use mp4 for higher fps")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_sequence_visualization({
        "config_file":  args.config_file,
        "model_file":   args.model_file,
        "base_path":    args.base_path,
        "output_dir":   args.output_dir,
        "num_clips":    args.num_clips,
        "start_clip":   args.start_clip,
        "clip_length":  args.clip_length,
        "fps":          args.fps,
        "format":       args.format,
    })


if __name__ == "__main__":
    main()
