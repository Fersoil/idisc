"""Compare the per-pixel discretization assignment across the three sam3_memory
5k configs: linear_proj, adapter (ISD cross-attention over SAM3 query IDRs) vs.
mask_pool (mask-derived assignment). For each pixel we measure how *concentrated*
the assignment over the K cluster slots is (max weight + normalized entropy):
high concentration = the model commits each pixel to a cluster (uses the
discretization); near-uniform = the IDR path is bypassed (depth comes from the FPN).

Runs one model at a time (3 SAM3 backbones do not co-fit in 16 GB), caches the
maps on CPU, then renders a combined figure per image. GPU node required.
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
from _viz_common import AttentionCapture, crop_to_content, extract_sample

import idisc.dataloders as custom_dataset
from idisc.models._sam3_common import denormalize_imagenet, letterbox_to_square
from idisc.models.idisc import IDisc

LB = 1008


def concentration(assign):  # assign: (hw, K) rows sum to 1 -> (maxw, norm_entropy) per pixel
    maxw = assign.max(-1).values
    ent = -(assign.clamp_min(1e-9) * assign.clamp_min(1e-9).log()).sum(-1)
    ent = ent / np.log(assign.shape[-1])
    return maxw, ent


def finest(d):
    return max(d, key=lambda k: d[k].shape[1])


def run_model(cfg_file, ckpt, sam_mode, imgs):
    config = OmegaConf.to_container(OmegaConf.load(cfg_file), resolve=True)
    nheads = config["model"]["num_heads"]
    model = IDisc.build(config).cuda().eval()
    model.load_pretrained(ckpt)

    seg = {}
    if sam_mode == "mask_pool":
        h = model.pixel_encoder.sam_model.segmentation_head.register_forward_hook(
            lambda m, i, o: seg.__setitem__("pm", o["pred_masks"][0].detach().float().cpu()))
    cap = AttentionCapture(model)

    out = {}
    with torch.no_grad():
        for idx, s in imgs.items():
            cap.reset(); seg.clear()
            model(s["image"][None].cuda(), sam_mode=sam_mode,
                  gt=s["gt"][None].cuda(), mask=s["mask"][None].cuda())
            if sam_mode == "mask_pool":
                hw = max(cap.isd_hw.values(), key=lambda t: t[0] * t[1])
                m = torch.sigmoid(F.interpolate(seg["pm"][None], size=hw,
                                  mode="bilinear", align_corners=False))[0]
                mf = m.flatten(1)                              # (K, hw)
                assign = (mf / (mf.sum(0, keepdim=True) + 1e-6)).T  # (hw, K)
            else:
                r = finest(cap.isd_attn)
                assign = extract_sample(cap.isd_attn[r], nheads, 0)  # (hw, K)
                hw = cap.isd_hw[r]
            maxw, ent = concentration(assign)
            out[idx] = (maxw.reshape(hw).numpy(), ent.reshape(hw).numpy())
    cap.remove()
    if sam_mode == "mask_pool":
        h.remove()
    del model
    torch.cuda.empty_cache()
    return out


def lb_rgb(img):
    return letterbox_to_square(denormalize_imagenet(img), size=LB).permute(1, 2, 0).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--out", default="docs/SAM2Depth/gifs_v2/assign")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs = [
        ("linear_proj", "output/runs/2026-06-01_11-37-54_lp_mem_e2e_b0dbe7e/resolved_config.yaml",
         "output/models/lp_mem_e2e/best_sam_finetuned.pt", "linear_proj"),
        ("adapter", "output/runs/2026-06-01_15-42-04_adp_mem_e2e_b0dbe7e/resolved_config.yaml",
         "output/models/adp_mem_e2e/best_sam_finetuned.pt", "adapter"),
        ("mask_pool", "output/runs/2026-06-02_13-13-50_maskpool_mem_e2e_a2abb79/resolved_config.yaml",
         "output/models/maskpool_mem_e2e/best_sam_finetuned.pt", "mask_pool"),
    ]

    c0 = OmegaConf.to_container(OmegaConf.load(runs[0][1]), resolve=True)
    ds = custom_dataset.KITTIDataset(
        test_mode=True,
        base_path=os.path.join(c0["paths"]["base_path"], c0["data"]["data_root"]),
        crop=c0["data"].get("crop"))
    step = max(1, len(ds) // args.n)
    idxs = list(range(0, args.n * step, step))
    imgs = {i: ds[i] for i in idxs}

    results = {}
    means = {}
    for name, cfg, ckpt, mode in runs:
        print(f"== {name} ==", flush=True)
        results[name] = run_model(cfg, ckpt, mode, imgs)
        means[name] = float(np.mean([results[name][i][0].mean() for i in idxs]))
        print(f"  mean max-assignment-weight = {means[name]:.3f}", flush=True)

    uniform = 1.0 / 200  # K=200 slots; a uniform assignment weighs 1/K per slot
    for k, i in enumerate(idxs, start=1):
        orig_hw = tuple(imgs[i]["image"].shape[-2:])
        square = lb_rgb(imgs[i]["image"])
        rgb = crop_to_content(square, orig_hw)   # KITTI band, not the padded square
        fig, ax = plt.subplots(1, 4, figsize=(22, 2.6))
        ax[0].imshow(rgb); ax[0].set_title("input")
        for j, (name, *_ ) in enumerate(runs):
            mw = results[name][i][0]
            mw = F.interpolate(torch.tensor(mw)[None, None], size=square.shape[:2],
                               mode="bilinear", align_corners=False)[0, 0].numpy()
            mw = crop_to_content(mw, orig_hw)
            im = ax[j + 1].imshow(mw, cmap="magma", vmin=0, vmax=0.1)
            ax[j + 1].set_title(f"{name} ({means[name] / uniform:.0f}×)")
        for a in ax:
            a.axis("off")
        # one shared bar across the three method panels: same scale → directly comparable
        cb = fig.colorbar(im, ax=ax[1:].tolist(), fraction=0.025, pad=0.01)
        cb.set_label("max assign weight")
        fig.suptitle("ISD per-pixel assignment concentration", fontsize=11, y=1.02)
        fig.savefig(os.path.join(args.out, f"assign_{k:02d}.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved assign_{k:02d}.png", flush=True)

    print("\nmean max-assignment-weight (higher = more concentrated discretization):")
    for name in means:
        print(f"  {name:12s} {means[name]:.3f}", flush=True)


if __name__ == "__main__":
    main()
