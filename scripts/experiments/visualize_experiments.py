#!/usr/bin/env python
"""
Visualize IDR attention maps from AFP and ISD cross-attention layers.

Uses forward hooks on each AttentionLayer's dropout sub-module to capture the
attention matrix (B*H, N, K) without modifying any model code.  A pre-hook on
AFP captures the feature-map spatial shapes needed to reshape K → H_feat × W_feat.

Outputs per-sample PNG grids saved to --output-dir.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.cuda as tcuda
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler

import idisc.dataloders as custom_dataset
from idisc.models.idisc import IDisc

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize_imagenet(img_tensor: torch.Tensor) -> np.ndarray:
    mean = IMAGENET_MEAN.to(img_tensor.device)
    std = IMAGENET_STD.to(img_tensor.device)
    return (img_tensor * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().float().numpy()


# ---------------------------------------------------------------------------
# Hook management
# ---------------------------------------------------------------------------

class AttentionCapture:
    """Registers forward hooks to collect attention matrices from AFP and ISD."""

    def __init__(self, model: IDisc) -> None:
        # afp_attn[res_idx]  -> (B*H, N_IDR, N_pixels)  last AFP iteration
        # afp_hw[res_idx]    -> (H_feat, W_feat)
        # isd_attn[res_idx]  -> (B*H, N_pixels, N_IDR)  last ISD depth layer
        # isd_hw[res_idx]    -> (H_feat, W_feat)
        self.afp_attn: Dict[int, torch.Tensor] = {}
        self.afp_hw: Dict[int, Tuple[int, int]] = {}
        self.isd_attn: Dict[int, torch.Tensor] = {}
        self.isd_hw: Dict[int, Tuple[int, int]] = {}
        self._handles: List = []
        self._register(model)

    def _register(self, model: IDisc) -> None:
        # --- AFP spatial shapes (pre-hook on AFP.forward) ---
        def _afp_pre(module, args):
            for idx, fm in enumerate(args[0]):
                self.afp_hw[idx] = (int(fm.shape[-2]), int(fm.shape[-1]))

        self._handles.append(model.afp.register_forward_pre_hook(_afp_pre))

        # --- AFP attention: hook dropout inside each cross_attn_{i+1}_d1 ---
        # The same layer is called `iters` times; overwriting keeps the last iteration.
        for i in range(model.afp.num_resolutions):
            ca = getattr(model.afp, f"cross_attn_{i+1}_d1")

            def _make_afp_hook(idx):
                def hook(module, inputs, outputs):
                    # inputs[0]: (B*H, N_IDR, N_pixels) — pre-dropout attention
                    self.afp_attn[idx] = inputs[0].detach().cpu()
                return hook

            self._handles.append(ca.dropout.register_forward_hook(_make_afp_hook(i)))

        # --- ISD spatial shapes + attention (last depth layer per head) ---
        for i in range(model.isd.num_resolutions):
            head = getattr(model.isd, f"head_{i+1}")

            def _make_isd_pre(idx):
                def hook(module, args):
                    fm = args[0]  # (B, C, H, W)
                    self.isd_hw[idx] = (int(fm.shape[-2]), int(fm.shape[-1]))
                return hook

            self._handles.append(head.register_forward_pre_hook(_make_isd_pre(i)))

            last_ca = getattr(head, f"cross_attn_{head.depth}")

            def _make_isd_hook(idx):
                def hook(module, inputs, outputs):
                    # inputs[0]: (B*H, N_pixels, N_IDR) — pre-dropout attention
                    self.isd_attn[idx] = inputs[0].detach().cpu()
                return hook

            self._handles.append(last_ca.dropout.register_forward_hook(_make_isd_hook(i)))

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def reset(self) -> None:
        self.afp_attn.clear()
        self.isd_attn.clear()


# ---------------------------------------------------------------------------
# Attention extraction helpers
# ---------------------------------------------------------------------------

def _extract_sample(
    attn_bh: torch.Tensor,
    num_heads: int,
    sample_idx: int,
) -> torch.Tensor:
    """
    Reshape (B*H, N, K) → (B, H, N, K), select sample, mean over heads → (N, K).

    Using [-1] on (B*H, N, K) gives the last head of the last sample, not the
    last sample.  This function handles the indexing correctly.
    """
    B = attn_bh.shape[0] // num_heads
    attn = attn_bh.view(B, num_heads, *attn_bh.shape[1:])  # (B, H, N, K)
    return attn[sample_idx].mean(0).float()                  # (N, K)


def _to_spatial(flat: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """(N, H*W) → (N, H, W) with bilinear upsample check."""
    return flat.reshape(flat.shape[0], h, w)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _save_idr_assignment(
    img_np: np.ndarray,
    assignment: torch.Tensor,   # (H_feat, W_feat) int — most-attending IDR per pixel
    num_idrs: int,
    title: str,
    path: Path,
) -> None:
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


def _save_idr_grid(
    img_np: np.ndarray,
    attn_hw: torch.Tensor,       # (N_IDR, H_feat, W_feat)
    title: str,
    path: Path,
    n_cols: int = 8,
) -> None:
    N_IDR = attn_hw.shape[0]
    n_rows = (N_IDR + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = np.array(axes).reshape(-1)

    for slot in range(N_IDR):
        ax = axes[slot]
        ax.imshow(img_np)
        heatmap = attn_hw[slot].numpy()
        ax.imshow(
            heatmap,
            alpha=0.55,
            cmap="hot",
            interpolation="bilinear",
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


def visualize_sample(
    image: torch.Tensor,
    capture: AttentionCapture,
    num_heads: int,
    sample_idx: int,
    output_dir: Path,
    tag: str,
) -> None:
    img_np = denormalize_imagenet(image[sample_idx])
    H_img, W_img = img_np.shape[:2]

    # --- AFP: IDRs attending to pixels ---
    for res_idx, attn_bh in capture.afp_attn.items():
        h_feat, w_feat = capture.afp_hw[res_idx]
        attn = _extract_sample(attn_bh, num_heads, sample_idx)  # (N_IDR, N_pixels)
        attn_hw = _to_spatial(attn, h_feat, w_feat)              # (N_IDR, H_feat, W_feat)

        # Upsample to image size for clean overlay
        attn_up = F.interpolate(
            attn_hw.unsqueeze(0), size=(H_img, W_img), mode="bilinear", align_corners=False
        ).squeeze(0)

        path = output_dir / f"{tag}_afp_res{res_idx + 1}_idr_attn.png"
        _save_idr_grid(
            img_np, attn_up,
            title=f"AFP IDR attention — res {res_idx + 1} — {tag}",
            path=path,
        )
        print(f"  saved {path.name}")

    # --- ISD: dominant IDR per pixel (clustering map) ---
    for res_idx, attn_bh in capture.isd_attn.items():
        h_feat, w_feat = capture.isd_hw[res_idx]
        # attn_bh: (B*H, N_pixels, N_IDR) — select sample, mean over heads → (N_pixels, N_IDR)
        attn = _extract_sample(attn_bh, num_heads, sample_idx)
        num_idrs = attn.shape[-1]

        # For each pixel, the IDR it attends to most
        assignment = attn.argmax(dim=-1).reshape(h_feat, w_feat)  # (H_feat, W_feat)

        path = output_dir / f"{tag}_isd_res{res_idx + 1}_idr_assignment.png"
        _save_idr_assignment(
            img_np, assignment, num_idrs,
            title=f"Dominant IDR per pixel — ISD res {res_idx + 1} — {tag}",
            path=path,
        )
        print(f"  saved {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_visualization(cfg: dict) -> None:
    with open(cfg["config_file"], "r", encoding="utf-8") as f:
        config = json.load(f)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda") if tcuda.is_available() else torch.device("cpu")
    num_heads = config["model"]["num_heads"]
    num_samples = cfg.get("num_samples", 8)

    model = IDisc.build(config)
    model.load_pretrained(cfg["model_file"])
    model = model.to(device).eval()
    print(f"Model loaded — {num_heads} heads, {model.afp.num_resolutions} AFP resolutions")

    data_path = os.path.join(cfg["base_path"], config["data"]["data_root"])
    dataset = getattr(custom_dataset, config["data"]["val_dataset"])(
        test_mode=True, base_path=data_path, crop=config["data"]["crop"]
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=SequentialSampler(dataset),
        num_workers=2,
        pin_memory=True,
        drop_last=False,
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

            model(data, gt=gt, mask=mask)

            visualize_sample(
                image=data,
                capture=capture,
                num_heads=num_heads,
                sample_idx=0,
                output_dir=output_dir,
                tag=f"sample_{i:04d}",
            )

    capture.remove()
    print(f"\nDone — {min(num_samples, len(dataset))} samples → {output_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize IDR attention maps")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--base-path", required=True)
    parser.add_argument("--output-dir", default="viz_results")
    parser.add_argument("--num-samples", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_visualization({
        "config_file": args.config_file,
        "model_file": args.model_file,
        "base_path": args.base_path,
        "output_dir": args.output_dir,
        "num_samples": args.num_samples,
    })


if __name__ == "__main__":
    main()
