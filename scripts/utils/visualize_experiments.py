#!/usr/bin/env python
"""Visualize IDR attention maps from AFP and ISD cross-attention layers.

Hooks each AttentionLayer's dropout sub-module to capture the attention matrix
(B*H, N, K) without touching model code; a pre-hook on AFP captures the
spatial shapes needed to reshape K → H_feat × W_feat. Per-sample PNG grids
are saved to --output-dir.
"""

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
from torch.utils.data import DataLoader, SequentialSampler

sys.path.insert(0, str(Path(__file__).parent))
from _viz_common import (
    AttentionCapture,
    SAM_MODE_CHOICES,
    denormalize_to_float01_hwc,
    extract_sample,
    to_spatial,
)

import idisc.dataloders as custom_dataset
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
    num_samples = cfg.get("num_samples", 8)
    sam_mode = cfg.get("sam_mode") or "replace"

    model = IDisc.build(config)
    model.load_pretrained(cfg["model_file"])
    model = model.to(device).eval()
    print(f"Model loaded — {num_heads} heads, {model.afp.num_resolutions} AFP resolutions")
    print(f"sam_mode: {sam_mode}")

    yields_iq = getattr(model.pixel_encoder, "yields_instance_queries", False)
    data_path = os.path.join(cfg["base_path"], config["data"]["data_root"])
    dataset = getattr(custom_dataset, config["data"]["val_dataset"])(
        test_mode=True, base_path=data_path, crop=config["data"]["crop"],
    )
    loader = DataLoader(
        dataset, batch_size=1, sampler=SequentialSampler(dataset),
        num_workers=2, pin_memory=True, drop_last=False,
    )
    print(f"{len(dataset)} samples — visualizing first {num_samples}")

    capture = AttentionCapture(model)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_samples:
                break
            data = batch["image"].to(device)
            gt = batch["gt"].to(device)
            mask = batch["mask"].to(device)
            capture.reset()

            # For SAM3 encoders, run the backbone once and pass pre-extracted
            # outputs to avoid a second backbone call inside model().
            if yields_iq:
                enc_out = model.pixel_encoder(data)
                *fpn, instance_queries = enc_out
                pre_extracted = model.invert_encoder_output_order(tuple(fpn))
            else:
                instance_queries = None
                pre_extracted = None

            model(data,
                  instance_queries=instance_queries,
                  sam_mode=sam_mode,
                  pre_extracted_encoder_outputs=pre_extracted,
                  gt=gt, mask=mask)

            visualize_sample(image=data, capture=capture, num_heads=num_heads,
                             sample_idx=0, output_dir=output_dir,
                             tag=f"sample_{i:04d}")

    capture.remove()
    print(f"\nDone — {min(num_samples, len(dataset))} samples → {output_dir}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize IDR attention maps")
    p.add_argument("--config-file", required=True,
                   help="Resolved Hydra config (resolved_config.yaml from a run dir).")
    p.add_argument("--model-file", required=True)
    p.add_argument("--base-path", required=True)
    p.add_argument("--output-dir", default="viz_results")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--sam-mode", default="replace", choices=SAM_MODE_CHOICES)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_visualization({
        "config_file": args.config_file,
        "model_file": args.model_file,
        "base_path": args.base_path,
        "output_dir": args.output_dir,
        "num_samples": args.num_samples,
        "sam_mode": args.sam_mode,
    })


if __name__ == "__main__":
    main()
