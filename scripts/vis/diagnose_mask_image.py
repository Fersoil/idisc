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
    m = IDisc.build(config).cuda().eval()
    m.load_pretrained(ckpt)
    return m


def masks_for(model, image):
    cap = {}
    h = model.pixel_encoder.sam_model.segmentation_head.register_forward_hook(
        lambda m, i, o: cap.__setitem__("pm", o["pred_masks"][0].detach().float().cpu()))
    with torch.no_grad():
        model.pixel_encoder(image.cuda())
    h.remove()
    return cap["pm"]


def stats(masks, size, orig_hw):
    prob = masks.sigmoid()
    up = F.interpolate(prob[None], size=size, mode="bilinear", align_corners=False)[0]
    t, l, h, w = content_box(orig_hw, up.shape[-2], up.shape[-1])
    up = up[:, t:t + h, l:l + w]
    maxp, arg = up.max(0)
    seg = torch.where(maxp > 0.5, arg, torch.full_like(arg, -1)).numpy()
    seg_full = arg.numpy()
    per_peak = prob.flatten(1).max(1).values
    per_area = (prob > 0.5).flatten(1).sum(1)
    active = per_area > 64

    act_idx = torch.where(active)[0]
    pa = up[act_idx] if len(act_idx) else up
    seg_a = pa.argmax(0)
    oh = F.one_hot(seg_a, pa.shape[0]).permute(2, 0, 1).float()
    vote = F.avg_pool2d(oh[None], 9, stride=1, padding=4)[0]
    seg_clean = vote.argmax(0).numpy()
    return dict(
        maxp=maxp.numpy(), seg=seg, seg_full=seg_full, seg_clean=seg_clean,
        n_active=int(active.sum()),
        coverage=float((maxp > 0.5).float().mean()),
        mean_peak=float(per_peak[active].mean()) if active.any() else 0.0,
        median_peak=float(per_peak.median()),
        sorted_peak=np.sort(per_peak.numpy())[::-1],
    )


def colorize(seg, n):
    rng = np.random.default_rng(0)
    pal = rng.uniform(0.25, 1.0, size=(n, 3))
    out = np.full((*seg.shape, 3), 0.5)
    for k in range(n):
        out[seg == k] = pal[k]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--orig", required=True)
    p.add_argument("--adapted", required=True)
    p.add_argument("--idx", type=int, default=216)
    p.add_argument("--out", default="docs/SAM2Depth/gifs_v2/diag")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    ds = custom_dataset.KITTIDataset(
        test_mode=True,
        base_path=os.path.join(config["paths"]["base_path"], config["data"]["data_root"]),
        crop=config["data"].get("crop"))
    img = ds[args.idx]["image"]
    orig_hw = tuple(img.shape[-2:])
    square = letterbox_to_square(denormalize_imagenet(img), size=LB).permute(1, 2, 0).cpu().numpy()
    size = square.shape[:2]
    rgb = crop_to_content(square, orig_hw)

    res = {}
    for tag, ckpt in [("original", args.orig), ("adapted", args.adapted)]:
        m = build(config, ckpt)
        mk = masks_for(m, img.unsqueeze(0))
        res[tag] = stats(mk, size, orig_hw); res[tag]["K"] = mk.shape[0]
        del m; torch.cuda.empty_cache()

    print(f"=== image idx {args.idx} ===", flush=True)
    for tag in ("original", "adapted"):
        s = res[tag]
        print(f"  {tag:9s}: active_masks={s['n_active']:3d}  coverage={s['coverage']:.2f}  "
              f"mean_peak_conf={s['mean_peak']:.3f}  median_peak(allK)={s['median_peak']:.3f}",
              flush=True)

    fig, ax = plt.subplots(2, 4, figsize=(22, 7))
    ax[0, 0].imshow(rgb); ax[0, 0].set_title("input")
    ax[1, 0].plot(res["original"]["sorted_peak"], label="original")
    ax[1, 0].plot(res["adapted"]["sorted_peak"], label="adapted")
    ax[1, 0].set_title("per-mask peak confidence (sorted)")
    ax[1, 0].set_xlabel("mask rank"); ax[1, 0].set_ylabel("peak sigmoid"); ax[1, 0].legend()
    ax[1, 0].axhline(0.5, ls="--", c="grey", lw=0.8)
    for r, tag in enumerate(("original", "adapted")):
        s = res[tag]
        ax[r, 1].imshow(colorize(s["seg_full"], s["K"]))
        ax[r, 1].set_title(f"{tag} full argmax (raw)")
        ax[r, 2].imshow(colorize(s["seg_clean"], int(s["seg_clean"].max()) + 1))
        ax[r, 2].set_title(f"{tag} filtered ({s['n_active']} active + mode)")
        im = ax[r, 3].imshow(s["maxp"], cmap="magma", vmin=0, vmax=1)
        ax[r, 3].set_title(f"{tag} max mask prob  (cov>0.5={s['coverage']:.2f})")
    for a in (ax[0, 1], ax[0, 2], ax[0, 3], ax[1, 1], ax[1, 2], ax[1, 3]):
        a.axis("off")
    ax[0, 0].axis("off")
    fig.colorbar(im, ax=ax[:, 3], fraction=0.04)
    fig.savefig(os.path.join(args.out, f"diag_{args.idx:04d}.png"), dpi=115, bbox_inches="tight")
    print(f"  saved {args.out}/diag_{args.idx:04d}.png", flush=True)


if __name__ == "__main__":
    main()
