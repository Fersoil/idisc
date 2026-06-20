import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))
from _viz_common import crop_to_content
from visualize_mask_pool import build, capture_masks, colorize, lb_image, partition

import idisc.dataloders as custom_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-multiclass", required=True)
    p.add_argument("--config-noprompt", required=True)
    p.add_argument("--ckpt-multiclass", required=True)
    p.add_argument("--ckpt-noprompt", required=True)
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--out", default="docs/SAM2Depth/gifs_v2/prompt")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg_m = OmegaConf.to_container(OmegaConf.load(args.config_multiclass), resolve=True)
    cfg_n = OmegaConf.to_container(OmegaConf.load(args.config_noprompt), resolve=True)
    data, paths = cfg_m["data"], cfg_m["paths"]
    ds = custom_dataset.KITTIDataset(
        test_mode=True,
        base_path=os.path.join(paths["base_path"], data["data_root"]),
        crop=data.get("crop"),
        manifest_path=data.get("manifest_path"),
    )

    m_multi = build(cfg_m, args.ckpt_multiclass)
    m_none = build(cfg_n, args.ckpt_noprompt)

    step = max(1, len(ds) // args.n)
    for k, i in enumerate(range(0, args.n * step, step), start=1):
        img = ds[i]["image"]
        orig_hw = tuple(img.shape[-2:])
        square = lb_image(img)
        size = square.shape[:2]
        rgb = crop_to_content(square, orig_hw)
        masks_m = capture_masks(m_multi, img.unsqueeze(0))
        masks_n = capture_masks(m_none, img.unsqueeze(0))
        seg_m, na_m, cov_m = partition(masks_m, size, orig_hw)
        seg_n, na_n, cov_n = partition(masks_n, size, orig_hw)

        fig, ax = plt.subplots(1, 3, figsize=(18, 2.6))
        ax[0].imshow(rgb); ax[0].set_title("input")
        ax[1].imshow(rgb); ax[1].imshow(colorize(seg_m, masks_m.shape[0]), alpha=0.55)
        ax[1].set_title(f"multiclass ({na_m} regions, {cov_m:.0%})")
        ax[2].imshow(rgb); ax[2].imshow(colorize(seg_n, masks_n.shape[0]), alpha=0.55)
        ax[2].set_title(f"no prompt ({na_n} regions, {cov_n:.0%})")
        for a in ax:
            a.axis("off")
        fig.suptitle("SAM3 mask partition: prompt vs. none (frozen)", fontsize=11, y=1.02)
        out_path = os.path.join(args.out, f"prompt_{k:02d}.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path}  (multiclass {na_m} vs no-prompt {na_n} regions)", flush=True)


if __name__ == "__main__":
    main()
