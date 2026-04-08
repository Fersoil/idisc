#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.utils.data.distributed
from torch.utils.data import DataLoader, SequentialSampler

import idisc.dataloders as custom_dataset
from idisc.models.idisc import IDisc


def main(config, args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = IDisc.build(config)
    model.load_pretrained(args.model_file)
    model = model.to(device)
    model.eval()

    f16 = config["training"].get("f16", False)
    context = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=f16)

    # Use the TEST dataset (test_mode=True, uses test_split)
    dataset_cls = getattr(custom_dataset, config["data"]["val_dataset"])
    dataset = dataset_cls(
        test_mode=True,
        base_path=config["data"]["data_root"],
        crop=config["data"].get("crop", None),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=4,
        sampler=SequentialSampler(dataset),
        pin_memory=True,
        drop_last=False,
    )

    out_dir = Path(args.pred_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running inference on {len(dataset)} samples → {out_dir}")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            image = batch["image"].to(device)
            with context:
                preds, losses, _ = model(image, None, None)
            depth = preds.squeeze().cpu().numpy().astype(np.float32)

            # Recover original filename index from dataset
            sample_info = dataset.dataset[i]
            img_path = sample_info["image_filename"]
            # e.g. ".../test_004519_rgb.png" → "004519"
            stem = Path(img_path).stem            # "test_004519_rgb"
            idx = stem.split("_")[1]              # "004519"

            out_path = out_dir / f"test_{idx}.npy"
            np.save(out_path, depth)

            if i % 100 == 0:
                print(f"  [{i}/{len(dataset)}] saved {out_path.name}")

    print(f"Done. Saved {len(dataset)} predictions to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--pred-dir", required=True, help="Where to save .npy predictions")
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = json.load(f)

    main(config, args)