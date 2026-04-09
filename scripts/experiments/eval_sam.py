#!/usr/bin/env python
"""
SAM3 detection evaluation on KITTI (D1-D4). No depth, just detection stats.

Prompt modes:
  none         No text prompt at all (just encoder features)
  singleclass  Per-class x8
  multiclass   One multi-class prompt
  classonly    Multi-class prompt, class-logit only (ignore presence)

Reports: time, detections per image, per-class stats, score distributions.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.cuda as tcuda
from torch.utils.data import DataLoader, SequentialSampler

import idisc.dataloders as custom_dataset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
MULTI_CLASS_PROMPT = "car . truck . person . bicycle . building . tree . road sign . pole"
KITTI_CLASSES = ["car", "truck", "person", "bicycle", "building", "tree", "road sign", "pole"]


def denormalize_imagenet(img_tensor):
    mean = IMAGENET_MEAN.to(img_tensor.device)
    std = IMAGENET_STD.to(img_tensor.device)
    img = img_tensor * std + mean
    return (img * 255).clamp(0, 255).byte()


class DetectionStats:
    def __init__(self):
        self.class_max = []
        self.presence_max = []
        self.combined_max = []
        self.n_above = {0.5: [], 0.3: [], 0.1: [], 0.05: []}
        self.per_class = {cls: {"n_det": [], "max_score": []} for cls in KITTI_CLASSES}

    def update(self, state, cls_name=None):
        diag = state.get("diagnostics")
        if diag is None:
            return
        cp = diag["class_probs"].float()
        pp = diag["presence_probs"].float()
        cb = diag["combined_probs"].float()
        self.class_max.append(cp.max().item())
        self.presence_max.append(pp.max().item())
        self.combined_max.append(cb.max().item())
        for t in self.n_above:
            self.n_above[t].append((cb > t).sum().item())
        if cls_name is not None:
            n_det = (cb > 0.5).sum().item()
            self.per_class[cls_name]["n_det"].append(n_det)
            self.per_class[cls_name]["max_score"].append(cb.max().item())

    def to_dict(self):
        result = {
            "class_prob_max": {"mean": float(np.mean(self.class_max)), "max": float(np.max(self.class_max))},
            "presence_max": {"mean": float(np.mean(self.presence_max)), "max": float(np.max(self.presence_max))},
            "combined_max": {"mean": float(np.mean(self.combined_max)), "max": float(np.max(self.combined_max))},
            "n_above_thresholds": {str(t): float(np.mean(v)) for t, v in self.n_above.items()},
        }
        per_class = {}
        for cls, vals in self.per_class.items():
            if vals["n_det"]:
                per_class[cls] = {
                    "mean_detections": float(np.mean(vals["n_det"])),
                    "mean_max_score": float(np.mean(vals["max_score"])),
                }
        if per_class:
            result["per_class"] = per_class
        return result

    def report(self):
        def _s(lst):
            a = np.array(lst)
            return f"mean={a.mean():.4f}  max={a.max():.4f}  min={a.min():.4f}"
        lines = [
            f"  class_prob  max : {_s(self.class_max)}",
            f"  presence    max : {_s(self.presence_max)}",
            f"  combined    max : {_s(self.combined_max)}",
        ]
        for t, vals in self.n_above.items():
            a = np.array(vals)
            lines.append(f"  detections > {t:<4}: mean={a.mean():.1f}  max={a.max():.0f}")
        for cls, vals in self.per_class.items():
            if vals["n_det"]:
                nd = np.array(vals["n_det"])
                ms = np.array(vals["max_score"])
                lines.append(f"  {cls:<12}: det={nd.mean():.2f}/img  max_score={ms.mean():.3f}")
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="SAM3 detection evaluation on KITTI")
    parser.add_argument("--prompt-mode", required=True,
                        choices=["none", "singleclass", "multiclass", "classonly"])
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--model-file", required=True,
                        help="iDisc model (only used to load the dataset config)")
    parser.add_argument("--base-path", required=True)
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--output-dir", default="eval_results")
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda") if tcuda.is_available() else torch.device("cpu")

    print(f"Prompt mode: {args.prompt_mode}", flush=True)
    print(f"Device:      {device}", flush=True)

    # Load data (just need images, no depth)
    data_path = os.path.join(args.base_path, config["data"]["data_root"])
    valid_dataset = getattr(custom_dataset, config["data"]["val_dataset"])(
        test_mode=True, base_path=data_path, crop=config["data"]["crop"])
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=2,
                              sampler=SequentialSampler(valid_dataset),
                              pin_memory=True, drop_last=False)
    print(f"{len(valid_dataset)} images.", flush=True)

    # Load SAM3
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    use_presence = (args.prompt_mode != "classonly")
    sam_model = build_sam3_image_model(
        device=str(device), checkpoint_path=args.sam_checkpoint,
        load_from_HF=False)
    sam_model.eval()
    proc = Sam3Processor(sam_model, device=str(device), use_presence_score=use_presence)
    print("SAM3 loaded.", flush=True)

    stats = DetectionStats()
    t0 = time.time()

    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            data = batch["image"].to(device)
            raw_img = denormalize_imagenet(data[0])

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = proc.set_image(raw_img)

                if args.prompt_mode == "none":
                    # No text prompt -- just run with dummy to get hidden states
                    proc.set_text_prompt(prompt="object", state=state)
                    stats.update(state)

                elif args.prompt_mode == "multiclass":
                    proc.set_text_prompt(prompt=MULTI_CLASS_PROMPT, state=state)
                    stats.update(state)

                elif args.prompt_mode == "classonly":
                    proc.set_text_prompt(prompt=MULTI_CLASS_PROMPT, state=state)
                    stats.update(state)

                elif args.prompt_mode == "singleclass":
                    for cls in KITTI_CLASSES:
                        proc.reset_all_prompts(state)
                        proc.set_text_prompt(prompt=cls, state=state)
                        stats.update(state, cls_name=cls)

            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(valid_loader)}", flush=True)

    elapsed = time.time() - t0

    print(f"\nDone in {elapsed:.1f}s ({elapsed/len(valid_dataset)*1000:.0f}ms/image)")
    print(f"{'='*50}")
    print(stats.report())
    print(f"{'='*50}")

    out_path = os.path.join(args.output_dir, "metrics.json")
    result = {
        "prompt_mode": args.prompt_mode,
        "n_images": len(valid_dataset),
        "elapsed_s": elapsed,
        "ms_per_image": elapsed / len(valid_dataset) * 1000,
        "detection": stats.to_dict(),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
