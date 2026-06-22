#!/usr/bin/env python

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.cuda as tcuda
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))
from _viz_common import (
    AttentionCapture,
    SAM_MODE_CHOICES,
    denormalize_to_float01_hwc,
    extract_sample,
    to_spatial,
)

from idisc.dataloders.kitti_sequence import KITTISequenceDataset
from idisc.models.idisc import IDisc


def _save_idr_assignment(img_np, assignment, num_idrs, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(img_np)
    axes[0].axis("off")
    axes[0].set_title("image")
    im = axes[1].matshow(assignment.numpy(), cmap="tab20", vmin=0, vmax=num_idrs - 1)
    axes[1].axis("off")
    axes[1].set_title("dominant IDR per pixel")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _save_idr_grid(img_np, attn_hw, title, path, n_cols=8):
    N_IDR = attn_hw.shape[0]
    n_rows = (N_IDR + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = np.array(axes).reshape(-1)
    for slot in range(N_IDR):
        ax = axes[slot]
        ax.imshow(img_np)
        ax.imshow(
            attn_hw[slot].numpy(),
            alpha=0.55, cmap="hot", interpolation="bilinear",
            extent=(0, img_np.shape[1], img_np.shape[0], 0),
        )
        ax.axis("off")
        ax.set_title(f"IDR {slot}", fontsize=6)
    for ax in axes[N_IDR:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def visualize_sample(image, capture, num_heads, sample_idx, output_dir, tag):
    img_np = denormalize_to_float01_hwc(image[sample_idx])
    H_img, W_img = img_np.shape[:2]

    for res_idx, attn_bh in capture.afp_attn.items():
        h_feat, w_feat = capture.afp_hw[res_idx]
        attn = extract_sample(attn_bh, num_heads, sample_idx)
        attn_hw = to_spatial(attn, h_feat, w_feat)
        attn_up = F.interpolate(
            attn_hw.unsqueeze(0), size=(H_img, W_img),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
        path = output_dir / f"{tag}_afp_res{res_idx + 1}_idr_attn.png"
        _save_idr_grid(img_np, attn_up,
                       title=f"AFP IDR attention — res {res_idx + 1} — {tag}",
                       path=path)
        print(f"  saved {path.name}")

    for res_idx, attn_bh in capture.isd_attn.items():
        h_feat, w_feat = capture.isd_hw[res_idx]
        attn = extract_sample(attn_bh, num_heads, sample_idx)
        num_idrs = attn.shape[-1]
        assignment = attn.argmax(dim=-1).reshape(h_feat, w_feat)
        path = output_dir / f"{tag}_isd_res{res_idx + 1}_idr_assignment.png"
        _save_idr_assignment(img_np, assignment, num_idrs,
                             title=f"Dominant IDR per pixel — ISD res {res_idx + 1} — {tag}",
                             path=path)
        print(f"  saved {path.name}")


def run_visualization(cfg: dict) -> None:
    config = OmegaConf.to_container(OmegaConf.load(cfg["config_file"]), resolve=True)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda") if tcuda.is_available() else torch.device("cpu")
    num_heads = config["model"]["num_heads"]
    num_clips = cfg.get("num_clips", 3)
    start_clip = cfg.get("start_clip", 0)
    sam_mode = cfg.get("sam_mode") or "replace"
    run_name = config.get("run", {}).get("name") or Path(cfg["config_file"]).stem

    model = IDisc.build(config)
    model.load_pretrained(cfg["model_file"])
    model = model.to(device).eval()
    if getattr(model.pixel_encoder, "is_video_encoder", False):
        raise RuntimeError(
            "This script is single-frame; the SAM3 video encoder needs a clip. "
            "Use visualize_sequence.py / visualize_sam3_masklets.py for the video model."
        )
    yields_iq = getattr(model.pixel_encoder, "yields_instance_queries", False)
    print(f"Model loaded ({run_name}) — {num_heads} heads, "
          f"{model.afp.num_resolutions} AFP resolutions, sam_mode={sam_mode}")
    if sam_mode == "replace" and yields_iq:
        print("note: replace mode bypasses AFP, so only ISD attention is captured "
              "(AFP grids appear for the AFP baseline only).")

    clip_length = cfg.get("clip_length") or config["data"].get("clip_length", 4)
    manifest = (cfg.get("manifest_path") or
                config["data"].get("manifest_path", "splits/kitti/sequence_manifest.json"))
    data_path = os.path.join(cfg["base_path"], config["data"]["data_root"])
    dataset = KITTISequenceDataset(
        test_mode=True, base_path=data_path,
        manifest_path=os.path.join(cfg["base_path"], manifest),
        clip_length=clip_length, crop=config["data"]["crop"],
    )
    if start_clip >= len(dataset):
        raise ValueError(f"--start-clip {start_clip} >= dataset size {len(dataset)}")
    end_clip = min(start_clip + num_clips, len(dataset))
    print(f"{len(dataset)} clips — frame 0 of clips {start_clip}..{end_clip - 1}")

    capture = AttentionCapture(model)
    with torch.no_grad():
        for clip_idx in range(start_clip, end_clip):
            clip = dataset[clip_idx]
            image = clip["images"][0:1].to(device)
            gt = clip["depths"][0:1].to(device)
            mask = clip["masks"][0:1].to(device)
            seq_id = clip["sequence_id"].replace("/", "_")
            capture.reset()

            if yields_iq:
                *fpn, instance_queries = model.pixel_encoder(image)
                pre_extracted = model.invert_encoder_output_order(tuple(fpn))
            else:
                instance_queries = None
                pre_extracted = None

            model(image,
                  instance_queries=instance_queries,
                  sam_mode=sam_mode,
                  pre_extracted_encoder_outputs=pre_extracted,
                  gt=gt, mask=mask)

            tag = f"clip_{clip_idx:03d}_{seq_id}"
            visualize_sample(image=image, capture=capture, num_heads=num_heads,
                             sample_idx=0, output_dir=output_dir, tag=tag)

    capture.remove()
    print(f"\nDone → {output_dir}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-IDR attention grids (AFP) + ISD dominant-IDR maps, single frame.")
    p.add_argument("--config-file", required=True,
                   help="Resolved Hydra config (resolved_config.yaml from a run dir).")
    p.add_argument("--model-file", required=True)
    p.add_argument("--base-path", required=True)
    p.add_argument("--output-dir", default="viz_attn")
    p.add_argument("--manifest-path", default=None)
    p.add_argument("--num-clips", type=int, default=3)
    p.add_argument("--start-clip", type=int, default=0)
    p.add_argument("--clip-length", type=int, default=None)
    p.add_argument("--sam-mode", default="replace", choices=SAM_MODE_CHOICES)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_visualization({
        "config_file": args.config_file,
        "model_file": args.model_file,
        "base_path": args.base_path,
        "output_dir": args.output_dir,
        "manifest_path": args.manifest_path,
        "num_clips": args.num_clips,
        "start_clip": args.start_clip,
        "clip_length": args.clip_length,
        "sam_mode": args.sam_mode,
    })


if __name__ == "__main__":
    main()
