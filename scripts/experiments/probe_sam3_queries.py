#!/usr/bin/env python
"""H2 diagnostic for E11-sam3-pure: measure how often the placeholder
query fires on real KITTI validation images, and the distribution of
SAM3 instance-query counts per image.

If the placeholder fires often (>20%), the d2c bottleneck is degenerate
on those samples and improving the prompt is more important than longer
training. If it's rare (<5%), longer training (H1) is the right next move.

Usage:
    python scripts/experiments/probe_sam3_queries.py \\
        --config-file configs/kitti/kitti_sam3.json \\
        --n-images 200
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="configs/kitti/kitti_sam3.json")
    parser.add_argument("--base-path", default="/work/courses/3dv/team17/idisc")
    parser.add_argument("--n-images", type=int, default=200)
    parser.add_argument("--prompt-mode", default=None,
                        help="Override prompt_mode (e.g. 'multiclass', 'empty')")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Diagnostic needs CUDA — SAM3 hardcodes device='cuda'.")
    device = torch.device("cuda")

    config_path = REPO_ROOT / args.config_file
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if args.prompt_mode is not None:
        config["model"]["pixel_encoder"]["prompt_mode"] = args.prompt_mode

    print(f"Config:       {config_path}")
    print(f"Prompt mode:  {config['model']['pixel_encoder']['prompt_mode']}")
    print(f"Classes:      {config['model']['pixel_encoder']['prompt_classes']}")
    print(f"Probing first {args.n_images} val images...\n")

    # Build only the encoder — no need for the full d2c head.
    from idisc.models.encoder import sam3_image
    enc_kwargs = {
        "img_size": config["model"]["pixel_encoder"]["img_size"],
        "sam_checkpoint": config["model"]["pixel_encoder"].get("sam_checkpoint"),
        "prompt_mode": config["model"]["pixel_encoder"]["prompt_mode"],
        "prompt_classes": config["model"]["pixel_encoder"].get("prompt_classes", []),
        "freeze_sam3": True,
        "load_from_HF": config["model"]["pixel_encoder"].get("load_from_HF", False),
        "use_presence_score": config["model"]["pixel_encoder"].get(
            "use_presence_score", True
        ),
        "top_k_queries": config["model"]["pixel_encoder"].get("top_k_queries", 32),
        "confidence_threshold": config["model"]["pixel_encoder"].get(
            "confidence_threshold", 0.0
        ),
    }
    encoder = sam3_image(**enc_kwargs).to(device).eval()

    # Set up KITTI val dataloader
    from idisc.dataloders import kitti as custom_kitti
    import os
    data_path = os.path.join(args.base_path, config["data"]["data_root"])
    val_dataset = custom_kitti.KITTIDataset(
        test_mode=True, base_path=data_path, crop=config["data"]["crop"],
    )
    print(f"Val dataset size: {len(val_dataset)}")
    n = min(args.n_images, len(val_dataset))

    # Detect placeholder by identity: encoder.placeholder_query is the
    # canonical zero (1, 256) tensor. We grab its data_ptr to detect it.
    placeholder_id = encoder.placeholder_query.data_ptr()

    counts = Counter()
    placeholder_hits = 0
    k_values = []
    failures = 0

    with torch.no_grad():
        for i in range(n):
            sample = val_dataset[i]
            image = sample["image"].unsqueeze(0).to(device)
            try:
                out = encoder(image)
            except Exception as exc:
                print(f"[{i}] FAILED: {type(exc).__name__}: {exc}")
                failures += 1
                continue
            *feats, queries = out
            # queries shape: (1, K, 256)
            k = int(queries.shape[1])
            k_values.append(k)
            counts[k] += 1
            # Detect placeholder: shape (1, 1, 256) AND value all zeros.
            if k == 1 and torch.all(queries == 0):
                placeholder_hits += 1
            if (i + 1) % 25 == 0:
                pct = 100 * placeholder_hits / (i + 1)
                avg_k = sum(k_values) / len(k_values)
                print(f"  [{i+1}/{n}] placeholder={placeholder_hits} "
                      f"({pct:.1f}%) | avg K={avg_k:.1f}")

    print()
    print("=" * 60)
    print(f"Total images probed: {n}")
    print(f"Failures:            {failures}")
    print(f"Placeholder hits:    {placeholder_hits} ({100*placeholder_hits/n:.1f}%)")
    print(f"K (queries per image) distribution:")
    if k_values:
        print(f"  min/median/mean/max: "
              f"{min(k_values)} / {sorted(k_values)[len(k_values)//2]} / "
              f"{sum(k_values)/len(k_values):.1f} / {max(k_values)}")
        print("  histogram (K → count):")
        for k in sorted(counts.keys()):
            bar = "█" * min(40, counts[k])
            print(f"    K={k:3d}  {counts[k]:4d}  {bar}")
    print("=" * 60)
    print()
    print("Decision rule:")
    print("  placeholder >20%:  prompt is the bottleneck → run H3 next")
    print("  placeholder <5%:   queries are flowing → run H1 (longer training)")
    print("  5-20%:             do both, but H3 first")


if __name__ == "__main__":
    main()
