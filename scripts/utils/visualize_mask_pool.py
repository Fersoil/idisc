"""Visualize the mask_pool discretization: SAM3's per-object masks that ISD uses
as cluster assignment. Compares the original (frozen SAM3) masks against the
depth-adapted masks (e2e checkpoint where the segmentation head trained under
depth loss). Renders the per-pixel dominant-mask partition over the letterboxed
input, so a change in the partition is visible directly.

Run on a GPU node (SAM3 hardcodes CUDA). Example:
  python scripts/utils/visualize_mask_pool.py \
    --config output/runs/<ts>_maskpool_mem_e2e_15k_*/resolved_config.yaml \
    --orig   output/models/maskpool_mem_frozen/best_sam_finetuned.pt \
    --adapted output/models/maskpool_mem_e2e_15k/best_sam_finetuned.pt \
    --n 6 --out docs/SAM2Depth/gifs_v2/mask_pool
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))
from _viz_common import content_box, crop_to_content

import idisc.dataloders as custom_dataset
from idisc.models._sam3_common import denormalize_imagenet, letterbox_to_square
from idisc.models.idisc import IDisc

LB = 1008


def build(config, ckpt):
    model = IDisc.build(config).cuda().eval()
    if ckpt:
        model.load_pretrained(ckpt)
    return model


def capture_masks(model, image):
    cap = {}

    def hook(_m, _i, out):
        cap["pm"] = out["pred_masks"].detach().float().cpu()

    h = model.pixel_encoder.sam_model.segmentation_head.register_forward_hook(hook)
    with torch.no_grad():
        model.pixel_encoder(image.cuda())
    h.remove()
    return cap["pm"][0]  # (K, Hm, Wm)


def partition(masks, size, orig_hw, min_area=64, max_slots=24):
    """Per-pixel dominant-mask map plus two readability stats: how many masks are
    actually active (area ≥ min_area) and the fraction of pixels any mask claims
    (coverage). Putting these in the titles makes the dense→coarse reorganisation
    legible without reading the prose. Cropped to the KITTI content band first, so
    the letterbox padding does not show as a square or dilute the coverage stat."""
    prob = masks.sigmoid()
    prob = F.interpolate(prob[None], size=size, mode="bilinear", align_corners=False)[0]
    t, l, h, w = content_box(orig_hw, prob.shape[-2], prob.shape[-1])
    prob = prob[:, t:t + h, l:l + w]
    areas = (prob > 0.5).flatten(1).sum(1)
    n_active = int((areas >= min_area).sum())
    keep = torch.argsort(areas, descending=True)[:max_slots]
    keep = keep[areas[keep] >= min_area]
    prob = prob[keep]
    if prob.numel() == 0:
        return np.full(size, -1), 0, 0.0
    maxp, arg = prob.max(0)
    coverage = float((maxp > 0.5).float().mean())
    seg = torch.where(maxp > 0.5, arg, torch.full_like(arg, -1))
    return seg.numpy(), n_active, coverage


def colorize(seg, n):
    rng = np.random.default_rng(0)
    palette = rng.uniform(0.25, 1.0, size=(n, 3))
    out = np.full((*seg.shape, 3), 0.5)  # background = grey
    for k in range(n):
        out[seg == k] = palette[k]
    return out


def lb_image(img):
    u8 = denormalize_imagenet(img)            # uint8 0..255
    sq = letterbox_to_square(u8, size=LB)     # uint8 square, matches encoder
    return sq.permute(1, 2, 0).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--orig", required=True)
    p.add_argument("--adapted", required=True)
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--out", default="docs/SAM2Depth/gifs_v2/mask_pool")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    data = config["data"]
    paths = config["paths"]
    ds = custom_dataset.KITTIDataset(
        test_mode=True,
        base_path=os.path.join(paths["base_path"], data["data_root"]),
        crop=data.get("crop"),
        manifest_path=data.get("manifest_path"),
    )

    m_orig = build(config, args.orig)
    m_adapt = build(config, args.adapted)

    step = max(1, len(ds) // args.n)
    for k, i in enumerate(range(0, args.n * step, step), start=1):
        sample = ds[i]
        img = sample["image"]
        orig_hw = tuple(img.shape[-2:])
        square = lb_image(img)
        size = square.shape[:2]
        rgb = crop_to_content(square, orig_hw)   # KITTI band, not the padded square
        masks_o = capture_masks(m_orig, img.unsqueeze(0))
        masks_a = capture_masks(m_adapt, img.unsqueeze(0))
        seg_o, na_o, cov_o = partition(masks_o, size, orig_hw)
        seg_a, na_a, cov_a = partition(masks_a, size, orig_hw)

        fig, ax = plt.subplots(1, 3, figsize=(18, 2.6))
        ax[0].imshow(rgb); ax[0].set_title("input")
        ax[1].imshow(rgb); ax[1].imshow(colorize(seg_o, masks_o.shape[0]), alpha=0.55)
        ax[1].set_title(f"frozen ({na_o} regions, {cov_o:.0%})")
        ax[2].imshow(rgb); ax[2].imshow(colorize(seg_a, masks_a.shape[0]), alpha=0.55)
        ax[2].set_title(f"depth-adapted 15k ({na_a} regions, {cov_a:.0%})")
        for a in ax:
            a.axis("off")
        fig.suptitle("SAM3 mask partition (ISD discretization)", fontsize=11, y=1.02)
        out_path = os.path.join(args.out, f"mask_vis_{k:02d}.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path}  (frozen {na_o} -> adapted {na_a} regions)", flush=True)


if __name__ == "__main__":
    main()
