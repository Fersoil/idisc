#!/usr/bin/env python
"""Visualize IDR attention maps over a KITTI sequence clip.

For each clip: per-frame depth (handling both image and video encoders),
captures AFP + ISD attention via forward hooks, saves per-frame PNGs and a
GIF animation with [RGB | depth | ISD dominant IDR per resolution].
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.cuda as tcuda
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))
from _viz_common import (
    AttentionCapture,
    ISD_LABELS,
    SAM_MODE_CHOICES,
    denormalize_to_float01_hwc,
    depth_cmap,
    extract_sample,
    idr_display,
    isd_assignments,
    make_writer,
)

from idisc.dataloders.kitti_sequence import KITTISequenceDataset
from idisc.models.idisc import IDisc


def _run_clip(model, images, depths, masks, capture,
              clip_idx, seq_id, sam_mode, num_heads):
    T = images.shape[0]
    encoder_is_video = getattr(model.pixel_encoder, "is_video_encoder", False)
    frames = []

    with torch.no_grad():
        if encoder_is_video:
            enc_out = model.pixel_encoder(images)
            fpn_levels = enc_out[:-1]
            queries_T = enc_out[-1]

        for t in range(T):
            capture.reset()
            frame = images[t:t + 1]
            gt = depths[t:t + 1]
            msk = masks[t:t + 1]

            if encoder_is_video:
                frame_fpn = tuple(lvl[t:t + 1] for lvl in fpn_levels)
                inv_frame_fpn = tuple(reversed(frame_fpn))
                pred, _, _ = model(
                    frame, instance_queries=queries_T[t], sam_mode=sam_mode,
                    pre_extracted_encoder_outputs=inv_frame_fpn,
                    gt=gt, mask=msk,
                )
            else:
                pred, _, _ = model(frame, sam_mode=sam_mode, gt=gt, mask=msk)

            for res_idx, attn_bh in capture.isd_attn.items():
                attn = extract_sample(attn_bh, num_heads, sample_idx=0)
                max_prob = attn.max(dim=-1).values.mean().item()
                entropy = -(attn * (attn + 1e-9).log()).sum(dim=-1).mean().item()
                print(f"  t={t} res{res_idx + 1}: max_prob={max_prob:.4f}  "
                      f"entropy={entropy:.3f}  N_IDR={attn.shape[-1]}")

            frames.append({
                "image": denormalize_to_float01_hwc(frame[0]),
                "depth": pred[0, 0].cpu().float().numpy(),
                "gt": gt[0, 0].cpu().float().numpy(),
                "clip_idx": clip_idx,
                "seq_id": seq_id,
                "isd_attn": {k: v.clone() for k, v in capture.isd_attn.items()},
                "isd_hw": dict(capture.isd_hw),
            })
    return frames


def _make_axes():
    fig, axes = plt.subplots(2, 3, figsize=(21, 7), constrained_layout=True)
    for ax in axes.flat:
        ax.axis("off")
    return fig, axes


def _init_images(axes, fd, num_heads, depth_max, num_idrs, cmap_depth, isd_label, soft):
    asgns = isd_assignments(fd, num_heads, soft=soft)
    cmap_idr, norm_idr, vmin_idr, vmax_idr = idr_display(num_idrs, soft)
    interp = "bilinear" if soft else "nearest"

    axes[0, 0].set_title("RGB", fontsize=12)
    im_rgb = axes[0, 0].imshow(fd["image"])
    axes[0, 1].set_title("Depth prediction", fontsize=12)
    im_d = axes[0, 1].imshow(fd["depth"], cmap=cmap_depth, vmin=0, vmax=depth_max)
    axes[0, 2].set_title("GT depth  (grey = no annotation)", fontsize=12)
    im_gt = axes[0, 2].imshow(
        np.ma.masked_equal(fd["gt"], 0), cmap=cmap_depth, vmin=0, vmax=depth_max,
    )

    im_asgns = []
    for col in range(3):
        if col in asgns:
            axes[1, col].set_title(f"{isd_label} — res {col + 1}", fontsize=12)
            kwargs = {"cmap": cmap_idr, "interpolation": interp}
            if norm_idr is not None:
                kwargs["norm"] = norm_idr
            else:
                kwargs["vmin"] = vmin_idr
                kwargs["vmax"] = vmax_idr
            im_asgns.append(axes[1, col].imshow(asgns[col], **kwargs))
    return {"rgb": im_rgb, "depth": im_d, "gt": im_gt, "asgns": im_asgns}


def _update_images(handles, fd, num_heads, soft):
    asgns = isd_assignments(fd, num_heads, soft=soft)
    handles["rgb"].set_data(fd["image"])
    handles["depth"].set_data(fd["depth"])
    handles["gt"].set_data(np.ma.masked_equal(fd["gt"], 0))
    for col, im in enumerate(handles["asgns"]):
        im.set_data(asgns[col])


def _add_idr_colorbar(fig, axes, im_asgns, num_idrs, soft):
    if not im_asgns:
        return
    cb = fig.colorbar(im_asgns[-1], ax=axes[1, 2], location="right", shrink=0.9)
    cb.set_label("mean IDR index" if soft else "IDR index")
    if not soft:
        step = max(1, num_idrs // 10)
        cb.set_ticks(range(0, num_idrs, step))


def _save_frame_png(fd, num_heads, depth_max, num_idrs, model_tag, isd_label, soft, path):
    fig, axes = _make_axes()
    handles = _init_images(axes, fd, num_heads, depth_max, num_idrs,
                           depth_cmap(), isd_label, soft)
    fig.colorbar(handles["depth"], ax=axes[0, 2], location="right",
                 shrink=0.9, label="depth (m)")
    _add_idr_colorbar(fig, axes, handles["asgns"], num_idrs, soft)
    fig.suptitle(
        f"clip {fd.get('clip_idx', '')}  {fd.get('seq_id', '')}\n{model_tag}",
        fontsize=11,
    )
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _save_gif(frames_data, num_heads, depth_max, num_idrs,
              model_tag, isd_label, soft, path, fps, fmt):
    fig, axes = _make_axes()
    fd0 = frames_data[0]
    handles = _init_images(axes, fd0, num_heads, depth_max, num_idrs,
                           depth_cmap(), isd_label, soft)
    fig.colorbar(handles["depth"], ax=axes[0, 2], location="right",
                 shrink=0.9, label="depth (m)")
    _add_idr_colorbar(fig, axes, handles["asgns"], num_idrs, soft)
    sup = fig.suptitle(
        f"clip {fd0.get('clip_idx', '')}  {fd0.get('seq_id', '')}  "
        f"— frame 1\n{model_tag}",
        fontsize=11,
    )

    def update(t):
        fd = frames_data[t]
        _update_images(handles, fd, num_heads, soft)
        sup.set_text(
            f"clip {fd.get('clip_idx', '')}  {fd.get('seq_id', '')}  "
            f"— frame {t + 1}/{len(frames_data)}\n{model_tag}"
        )
        return []

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames_data), interval=1000 // fps, blit=False,
    )
    anim.save(path, writer=make_writer(fmt, fps))
    plt.close(fig)


def run_sequence_visualization(cfg: dict) -> None:
    config = OmegaConf.to_container(OmegaConf.load(cfg["config_file"]), resolve=True)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda") if tcuda.is_available() else torch.device("cpu")
    num_heads = config["model"]["num_heads"]
    num_clips = cfg.get("num_clips", 3)
    start_clip = cfg.get("start_clip", 0)
    fps = cfg.get("fps", 2)
    fmt = cfg.get("format", "gif")
    sam_mode = cfg.get("sam_mode") or "replace"
    soft = cfg.get("soft_assignment", False)

    model = IDisc.build(config)
    model.load_pretrained(cfg["model_file"])
    model = model.to(device).eval()
    print(f"Model loaded — {num_heads} heads, {model.afp.num_resolutions} AFP resolutions")

    run_name = config.get("run", {}).get("name") or Path(cfg["config_file"]).stem
    idr_source = config.get("method", {}).get("idr_source")
    model_tag = f"{run_name}  sam_mode={sam_mode}"
    if idr_source == "afp":
        isd_label = "AFP IDR assignment"
    else:
        isd_label = ISD_LABELS.get(sam_mode, "IDR assignment")

    clip_length = cfg.get("clip_length") or config["data"].get("clip_length", 4)
    data_path = os.path.join(cfg["base_path"], config["data"]["data_root"])
    manifest = (cfg.get("manifest_path") or
                config["data"].get("manifest_path", "splits/kitti/sequence_manifest.json"))
    dataset = KITTISequenceDataset(
        test_mode=True, base_path=data_path,
        manifest_path=os.path.join(cfg["base_path"], manifest),
        clip_length=clip_length, crop=config["data"]["crop"],
    )
    print(f"{len(dataset)} clips — visualizing first {num_clips}")

    capture = AttentionCapture(model)
    if start_clip >= len(dataset):
        raise ValueError(
            f"--start-clip {start_clip} out of range (dataset has {len(dataset)} clips)"
        )
    end_clip = min(start_clip + num_clips, len(dataset))

    for clip_idx in range(start_clip, end_clip):
        clip = dataset[clip_idx]
        images = clip["images"].to(device)
        depths = clip["depths"].to(device)
        masks = clip["masks"].to(device)
        seq_id = clip["sequence_id"].replace("/", "_")
        T = images.shape[0]
        print(f"\nClip {clip_idx}  seq={seq_id}  frames={clip['frame_indices']}")
        frames_data = _run_clip(model, images, depths, masks, capture,
                                clip_idx, seq_id, sam_mode, num_heads)
        print(f"  {T} frames collected")

        depth_max = max(fd["depth"].max() for fd in frames_data)
        first_res = min(frames_data[0]["isd_attn"].keys())
        num_idrs = extract_sample(
            frames_data[0]["isd_attn"][first_res], num_heads, 0,
        ).shape[-1]

        clip_dir = output_dir / f"clip_{clip_idx:03d}_{seq_id}"
        clip_dir.mkdir(exist_ok=True)
        for t, fd in enumerate(frames_data):
            png_path = clip_dir / f"frame_{t:03d}.png"
            _save_frame_png(fd, num_heads, depth_max, num_idrs,
                            model_tag, isd_label, soft, png_path)
            print(f"  saved {png_path.name}")

        suffix = f"{sam_mode}_{'soft' if soft else 'hard'}"
        anim_path = output_dir / f"clip_{clip_idx:03d}_{seq_id}_{suffix}.{fmt}"
        _save_gif(frames_data, num_heads, depth_max, num_idrs,
                  model_tag, isd_label, soft, anim_path, fps, fmt)
        print(f"  saved {anim_path.name}")

    capture.remove()
    print(f"\nDone → {output_dir}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize IDR attention over KITTI sequences")
    p.add_argument("--config-file", required=True,
                   help="Resolved Hydra config (resolved_config.yaml).")
    p.add_argument("--model-file", required=True)
    p.add_argument("--base-path", required=True)
    p.add_argument("--output-dir", default="viz_sequence")
    p.add_argument("--num-clips", type=int, default=3)
    p.add_argument("--start-clip", type=int, default=0)
    p.add_argument("--manifest-path", default=None)
    p.add_argument("--clip-length", type=int, default=None)
    p.add_argument("--sam-mode", default="replace", choices=SAM_MODE_CHOICES)
    p.add_argument("--soft-assignment", action="store_true",
                   help="Weighted-average IDR index (continuous viridis) instead "
                        "of hard argmax. Stable across frames.")
    p.add_argument("--fps", type=int, default=2)
    p.add_argument("--format", default="gif", choices=["gif", "mp4"])
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_sequence_visualization({
        "config_file": args.config_file,
        "model_file": args.model_file,
        "base_path": args.base_path,
        "output_dir": args.output_dir,
        "num_clips": args.num_clips,
        "start_clip": args.start_clip,
        "sam_mode": args.sam_mode,
        "soft_assignment": args.soft_assignment,
        "manifest_path": args.manifest_path,
        "clip_length": args.clip_length,
        "fps": args.fps,
        "format": args.format,
    })


if __name__ == "__main__":
    main()
