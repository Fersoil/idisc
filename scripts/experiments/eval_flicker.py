#!/usr/bin/env python
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, SequentialSampler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import idisc.dataloders as custom_dataset
from idisc.models.idisc import IDisc
from idisc.models.sam3_track import Sam3TrackModule
from idisc.optimization.grounding_losses import temporal_smoothness_loss


def run(cfg, checkpoint_path, output_dir, limit):
    if cfg["run"]["dataset_mode"] != "video":
        raise ValueError(f"flicker eval expects dataset_mode=video, got {cfg['run']['dataset_mode']!r}")
    tcfg = cfg["finetune"]["temporal"]
    sam_mode = cfg["method"]["sam_mode"]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(output_dir, exist_ok=True)

    model = IDisc.build(cfg).to(device)
    model.load_pretrained(checkpoint_path)
    model.eval()
    track = Sam3TrackModule(
        sam_checkpoint=tcfg["sam_checkpoint"],
        prompt_classes=tcfg["prompt"]["classes"],
        confidence_threshold=tcfg["confidence_threshold"],
    ).to(device)

    dataset_cls = custom_dataset.select_dataset_cls(cfg["dataset_name"], "video")
    data_path = os.path.join(cfg["paths"]["base_path"], cfg["data"]["data_root"])
    dataset = getattr(custom_dataset, dataset_cls)(
        test_mode=True, base_path=data_path,
        manifest_path=cfg["data"]["manifest_path"],
        clip_length=cfg["data"]["clip_length"], crop=cfg["data"]["crop"],
    )
    loader = DataLoader(
        dataset, batch_size=1, num_workers=2,
        sampler=SequentialSampler(dataset), pin_memory=True,
    )
    print(f"{len(dataset)} clips | classes={tcfg['prompt']['classes']}", flush=True)

    flick = []
    t0 = time.time()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if limit and i >= limit:
                break
            images = batch["images"].to(device)
            B, T = images.shape[:2]
            for b in range(B):
                clip = images[b]
                labels = track(clip)
                preds = [model(clip[t:t + 1], sam_mode=sam_mode)[0] for t in range(T)]
                flick.append(temporal_smoothness_loss(torch.cat(preds, dim=0), labels).item())
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(loader)}", flush=True)

    if not flick:
        raise RuntimeError("no clips scored")
    result = {
        "checkpoint": checkpoint_path,
        "flicker": float(np.mean(flick)),
        "n_clips": len(flick),
        "elapsed_s": time.time() - t0,
    }
    out_path = os.path.join(output_dir, "flicker.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"Saved to {out_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Per-instance temporal flicker on KITTI clips")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="output/runs/flicker")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    run(cfg, args.checkpoint, args.output_dir, args.limit)


if __name__ == "__main__":
    main()
