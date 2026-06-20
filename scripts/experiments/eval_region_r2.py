#!/usr/bin/env python
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.experiments._eval_common import build_eval_model, build_val_loader
from idisc.models.sam3_masks import crop_letterbox_to_frame


def region_r2(mask_logits, gt, valid, tau):
    probs = mask_logits.sigmoid()
    conf = probs.amax(0)
    assign = probs.argmax(0)
    sel = valid & (conf > tau)
    if sel.sum() < 2:
        return None, 0
    d = gt[sel]
    a = assign[sel]
    ss_tot = ((d - d.mean()) ** 2).sum()
    if ss_tot <= 0:
        return None, 0
    uniq, inv = torch.unique(a, return_inverse=True)
    sums = torch.zeros(uniq.numel(), device=d.device).scatter_add(0, inv, d)
    cnts = torch.zeros(uniq.numel(), device=d.device).scatter_add(0, inv, torch.ones_like(d))
    region_mean = (sums / cnts)[inv]
    ss_res = ((d - region_mean) ** 2).sum()
    return float(1.0 - ss_res / ss_tot), int(uniq.numel())


def run(cfg, checkpoint_path, output_dir, tau, limit):
    if cfg["run"]["dataset_mode"] != "image":
        raise ValueError(f"region R2 expects dataset_mode=image, got {cfg['run']['dataset_mode']!r}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(output_dir, exist_ok=True)

    model = build_eval_model(cfg, checkpoint_path, device)
    if not getattr(model.pixel_encoder, "yields_instance_masks", False):
        raise ValueError(
            f"sam_mode={cfg['method']['sam_mode']!r} does not yield SAM3 masks; "
            "region R2 needs a mask_pool/mask_linear/mask_adapter checkpoint"
        )
    letterbox_size = model.pixel_encoder.letterbox_size
    loader = build_val_loader(cfg)
    print(f"{len(loader)} samples | tau={tau} | letterbox={letterbox_size}", flush=True)

    r2s, actives = [], []
    t0 = time.time()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if limit and i >= limit:
                break
            image = batch["image"].to(device)
            gt = batch["gt"].to(device)[0, 0]
            valid = batch["mask"].to(device).bool()[0, 0] & (gt > 0)
            mask_logits = model.pixel_encoder(image)[-1]
            mask_logits = crop_letterbox_to_frame(
                mask_logits.float(), tuple(gt.shape), letterbox_size
            )[0]
            r2, active = region_r2(mask_logits, gt, valid, tau)
            if r2 is not None:
                r2s.append(r2)
                actives.append(active)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(loader)}", flush=True)

    if not r2s:
        raise RuntimeError("no images scored; check masks and ground-truth validity")
    r2_t = torch.tensor(r2s)
    result = {
        "checkpoint": checkpoint_path,
        "tau": tau,
        "mean_r2": float(r2_t.mean()),
        "median_r2": float(r2_t.median()),
        "n_images_scored": len(r2s),
        "mean_active_masks": float(torch.tensor(actives, dtype=torch.float32).mean()),
        "elapsed_s": time.time() - t0,
    }
    out_path = os.path.join(output_dir, "region_r2.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"Saved to {out_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Region/depth coherence R2 for SAM3 masks")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="output/runs/region-r2")
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    run(cfg, args.checkpoint, args.output_dir, args.tau, args.limit)


if __name__ == "__main__":
    main()
