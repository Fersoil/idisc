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
from idisc.utils import DICT_METRICS_DEPTH, RunningMetric


def run_eval(cfg: dict, checkpoint_path: str, output_dir: str) -> dict:
    sam_mode = cfg["method"]["sam_mode"]
    dataset_mode = cfg["run"]["dataset_mode"]
    encoder_name = cfg["model"]["pixel_encoder"]["name"]
    encoder_is_video = encoder_name == "sam3_video"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Device: {device} | encoder={encoder_name} | sam_mode={sam_mode}", flush=True)

    model = build_eval_model(cfg, checkpoint_path, device)
    print(f"Loaded checkpoint: {checkpoint_path}", flush=True)
    valid_loader = build_val_loader(cfg)
    print(f"{len(valid_loader)} samples.", flush=True)

    tracker = RunningMetric(list(DICT_METRICS_DEPTH.keys()))
    t0 = time.time()
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            if encoder_is_video and "images" in batch:
                images = batch["images"].to(device)
                depths = batch["depths"].to(device)
                masks = batch["masks"].to(device)
                B, T = images.shape[:2]
                preds, vm, vd = [], [], []
                for b in range(B):
                    enc_out = model.pixel_encoder(images[b])
                    fpn_levels, queries_T = enc_out[:-1], enc_out[-1]
                    for t in range(T):
                        if masks[b, t].bool().sum() == 0:
                            continue
                        frame_fpn = tuple(lvl[t:t + 1] for lvl in fpn_levels)
                        pred, _, _ = model(
                            images[b, t:t + 1],
                            instance_queries=queries_T[t],
                            sam_mode=sam_mode,
                            pre_extracted_encoder_outputs=tuple(reversed(frame_fpn)),
                            gt=depths[b, t:t + 1],
                            mask=masks[b, t:t + 1],
                        )
                        preds.append(pred)
                        vm.append(masks[b, t:t + 1])
                        vd.append(depths[b, t:t + 1])
                if preds:
                    tracker.accumulate_metrics(
                        torch.cat(vd, dim=0).permute(0, 2, 3, 1),
                        torch.cat(preds, dim=0).permute(0, 2, 3, 1),
                        torch.cat(vm, dim=0).permute(0, 2, 3, 1),
                    )
            else:
                data = batch["image"].to(device) if "image" in batch \
                    else batch["images"].to(device).view(-1, *batch["images"].shape[2:])
                gt = batch["gt"].to(device) if "gt" in batch \
                    else batch["depths"].to(device).view(-1, *batch["depths"].shape[2:])
                mask = batch["mask"].to(device) if "mask" in batch \
                    else batch["masks"].to(device).view(-1, *batch["masks"].shape[2:])
                preds = []
                for idx in range(data.shape[0]):
                    pred, _, _ = model(
                        data[idx:idx + 1],
                        sam_mode=sam_mode,
                        gt=gt[idx:idx + 1],
                        mask=mask[idx:idx + 1],
                    )
                    preds.append(pred)
                tracker.accumulate_metrics(
                    gt.permute(0, 2, 3, 1),
                    torch.cat(preds, dim=0).permute(0, 2, 3, 1),
                    mask.permute(0, 2, 3, 1),
                )
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(valid_loader)}", flush=True)

    elapsed = time.time() - t0
    metrics = tracker.get_metrics()
    print(f"\nDone in {elapsed:.1f}s")
    for k, v in sorted(metrics.items()):
        print(f"  {k:<12} {v:.6f}")

    out_path = os.path.join(output_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump({"metrics": metrics, "elapsed_s": elapsed}, f, indent=2)
    print(f"Saved to {out_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Standalone depth evaluation")
    parser.add_argument("--config", required=True,
                        help="Path to resolved_config.yaml (saved by run_with_hydra)")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--output-dir", default="eval_results")
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    run_eval(cfg, args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
