#!/usr/bin/env python
"""
Depth evaluation for all E1-E10 experiment variants.
Outputs metrics to stdout and saves metrics.json.

Variants:
  baseline          E1: AFP only, no SAM3
  pooled            E2-E4: avg_pool2d on raw encoder hidden states (SAM3 doesn't affect depth)
  sam-replace       E5-E6: linear projection replaces AFP
  sam-concat        E7-E9: linear projection concatenated with AFP
  sam-cached-video  E10: concat with cached video queries

Prompt modes (for pooled/replace/concat):
  empty        empty string ""
  multiclass   "car . truck . person . bicycle . building . tree . road sign . pole"
  singleclass  per-class x8, merge top-32
  classonly    multi-class prompt but class-logit only (no presence)
"""

import argparse
import json
import os
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.cuda as tcuda
from torch.utils.data import DataLoader, SequentialSampler

import idisc.dataloders as custom_dataset
from idisc.models.idisc import IDisc
from idisc.utils import DICT_METRICS_DEPTH, RunningMetric

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
MULTI_CLASS_PROMPT = "car . truck . person . bicycle . building . tree . road sign . pole"
KITTI_CLASSES = ["car", "truck", "person", "bicycle", "building", "tree", "road sign", "pole"]


def denormalize_imagenet(img_tensor):
    mean = IMAGENET_MEAN.to(img_tensor.device)
    std = IMAGENET_STD.to(img_tensor.device)
    img = img_tensor * std + mean
    return (img * 255).clamp(0, 255).byte()


# ---------------------------------------------------------------------------
# SAM3 query extraction
# ---------------------------------------------------------------------------

def get_sam_queries_pooled(processor, raw_img, prompt_mode):
    """Pooled path: average encoder hidden states over layers, then avg_pool2d to (32, 128)."""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = processor.set_image(raw_img)
        if prompt_mode == "empty":
            processor.set_text_prompt(prompt="", state=state)
        elif prompt_mode == "multiclass":
            processor.set_text_prompt(prompt=MULTI_CLASS_PROMPT, state=state)
        elif prompt_mode == "singleclass":
            for cls in KITTI_CLASSES:
                processor.reset_all_prompts(state)
                processor.set_text_prompt(prompt=cls, state=state)
        else:
            processor.set_text_prompt(prompt=MULTI_CLASS_PROMPT, state=state)

    hs = state.get("hidden_states")
    if hs is None:
        return None
    # hs: (num_layers, batch, num_queries, 256)
    x = torch.mean(hs, dim=0)  # mean over layers -> (batch, num_queries, 256)
    x = F.adaptive_avg_pool2d(x, (32, 128)).squeeze(0)  # (32, 128)
    raw_idrs = (x.unsqueeze(0), x.clone().unsqueeze(0), x.clone().unsqueeze(0))
    return raw_idrs


def get_sam_queries_proj(processor, raw_img, prompt_mode):
    """Linear projection path: extract top-32 instance queries (32, 256)."""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = processor.set_image(raw_img)

        if prompt_mode in ("multiclass", "classonly"):
            processor.set_text_prompt(prompt=MULTI_CLASS_PROMPT, state=state)
            iq = state.get("instance_queries")
            return iq.float().clone() if iq is not None else None

        elif prompt_mode == "singleclass":
            all_queries = []
            for cls in KITTI_CLASSES:
                processor.reset_all_prompts(state)
                processor.set_text_prompt(prompt=cls, state=state)
                iq = state.get("instance_queries")
                if iq is not None and iq.shape[0] > 0:
                    all_queries.append(iq)
            if not all_queries:
                return None
            return torch.cat(all_queries, dim=0).float().clone()

        else:  # empty
            processor.set_text_prompt(prompt="", state=state)
            iq = state.get("instance_queries")
            return iq.float().clone() if iq is not None else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _validate_eval_args(cfg: dict[str, Any]) -> None:
    variant = cfg["variant"]
    config = cfg.get("config")
    encoder_owns_sam3 = (
        config is not None
        and config.get("model", {}).get("pixel_encoder", {}).get("name") == "sam3_image"
    )
    if (
        variant in ("pooled", "sam-replace", "sam-concat")
        and not encoder_owns_sam3
        and cfg.get("sam_checkpoint") is None
    ):
        raise ValueError(f"sam_checkpoint is required for variant '{variant}'")
def main():
    parser = argparse.ArgumentParser(description="Depth evaluation")
    parser.add_argument("--variant", required=True,
                        choices=["baseline", "pooled", "sam-replace", "sam-concat"])
    parser.add_argument("--prompt-mode", default="multiclass",
                        choices=["empty", "multiclass", "singleclass", "classonly"])
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--base-path", required=True)
    parser.add_argument("--sam-checkpoint", default=None)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()


def visualize (attn_matrix, img, name, save_path):
    import matplotlib.pyplot as plt
    import math
    import numpy as np
    import matplotlib as mpl

    if len(attn_matrix.shape) == 3:
        attn_matrix.squeeze()
    num_idrs = attn_matrix.shape[-1]

    most_attended_idr = torch.argmax(attn_matrix, dim=-1).squeeze()
    assert(len(most_attended_idr.shape) == 1)

    num_pixels = len(most_attended_idr)
    h, w = img.shape[-2], img.shape[-1]
    scale = int(math.sqrt(h*w / num_pixels))
    most_attended_idr = most_attended_idr.reshape((h // scale, w // scale))

    fig, axs = plt.subplots(2, 1, figsize=(6, 6))
    
    axs[0].imshow(img.cpu().numpy().squeeze().transpose(1,2,0)) 
    axs[1].matshow(most_attended_idr.cpu(), cmap='viridis', vmin=0, vmax=num_idrs-1)
    plt.savefig(os.path.join(save_path, "figs", name))


def run_eval(cfg: dict[str, Any]) -> dict[str, float]:
    variant = cfg["variant"]
    prompt_mode = cfg.get("prompt_mode", "multiclass")

    _validate_eval_args(cfg)

    config = cfg.get("config")
    if config is None:
        with open(cfg["config_file"], "r", encoding="utf-8") as f:
            config = json.load(f)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs(os.path.join(cfg["output_dir"], "figs"), exist_ok=True)
    device = torch.device("cuda") if tcuda.is_available() else torch.device("cpu")

    print(f"Variant:    {variant}", flush=True)
    print(f"Prompt:     {prompt_mode}", flush=True)
    print(f"Device:     {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU:        {torch.cuda.get_device_name(0)}", flush=True)

    encoder_owns_sam3 = (
        config["model"]["pixel_encoder"].get("name") == "sam3_image"
    )

    # Load iDisc
    model = IDisc.build(config)
    if cfg.get("load_pretrained", not encoder_owns_sam3):
        model.load_pretrained(cfg["model_file"])
        print("iDisc loaded (pretrained).", flush=True)
    else:
        print("iDisc built from scratch (no pretrained weights).", flush=True)
    model = model.to(device)
    model.eval()

    # Load data
    data_path = os.path.join(cfg["base_path"], config["data"]["data_root"])
    valid_dataset = getattr(custom_dataset, config["data"]["val_dataset"])(
        test_mode=True, base_path=data_path, crop=config["data"]["crop"])
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=2,
                              sampler=SequentialSampler(valid_dataset),
                              pin_memory=True, drop_last=False)
    print(f"{len(valid_dataset)} samples.", flush=True)

    f16 = config["training"].get("f16", False)
    context = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=f16)

    # Load SAM3 if needed (skipped when the pixel_encoder owns it)
    sam_proc = None
    if variant in ("pooled", "sam-replace", "sam-concat") and not encoder_owns_sam3:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        use_presence = (prompt_mode != "classonly")
        sam_model = build_sam3_image_model(
            device=str(device), checkpoint_path=cfg.get("sam_checkpoint"),
            load_from_HF=(cfg.get("sam_checkpoint") is None))
        sam_model.eval()
        sam_proc = Sam3Processor(sam_model, device=str(device), use_presence_score=use_presence)
        print("SAM3 loaded.", flush=True)

    # Run eval
    metrics_tracker = RunningMetric(list(DICT_METRICS_DEPTH.keys()))
    t0 = time.time()

    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            data = batch["image"].to(device)
            gt = batch["gt"].to(device)
            mask = batch["mask"].to(device)

            # Determine what to pass to model
            instance_queries = None
            raw_idrs = None
            sam_mode = "concat"

            if variant == "baseline":
                pass  # just AFP

            elif variant == "pooled":
                raw_img = denormalize_imagenet(data[0])
                raw_idrs = get_sam_queries_pooled(sam_proc, raw_img, prompt_mode)
                if raw_idrs is not None:
                    raw_idrs = tuple(r.to(device).float() for r in raw_idrs)

            elif variant == "sam-replace":
                if sam_proc is not None:
                    raw_img = denormalize_imagenet(data[0])
                    instance_queries = get_sam_queries_proj(sam_proc, raw_img, prompt_mode)
                    if instance_queries is not None:
                        instance_queries = instance_queries.to(device)
                # else: encoder owns SAM3 and yields queries from its forward
                sam_mode = "replace"

            elif variant == "sam-concat":
                if sam_proc is not None:
                    raw_img = denormalize_imagenet(data[0])
                    instance_queries = get_sam_queries_proj(sam_proc, raw_img, prompt_mode)
                    if instance_queries is not None:
                        instance_queries = instance_queries.to(device)
                sam_mode = "concat"

            with context:
                pred, _, meta = model(data,
                                   instance_queries=instance_queries,
                                   raw_idrs=raw_idrs,
                                   sam_mode=sam_mode,
                                   gt=gt, mask=mask)
                                   
                idrs = meta["queries"]
                
                # Visualize pixel to IDR attention from the batch at all resolutions for the last image of the batch
                for res_num in range(model.num_resolutions):
                    head = getattr(model.isd, f"head_{res_num+1}")
                    head_depth = head.depth
                    attn_module = getattr(head, f"cross_attn_{head_depth}")
                    attn_matrix = getattr(attn_module, "attn")
                    
                    visualize(attn_matrix.clone(), data[-1].clone(), save_path=cfg["output_dir"], name=f"{i:03}_{res_num+1}.png")


            metrics_tracker.accumulate_metrics(
                gt.permute(0, 2, 3, 1), pred.permute(0, 2, 3, 1), mask.permute(0, 2, 3, 1))
     
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(valid_loader)}", flush=True)

    elapsed = time.time() - t0
    metrics = metrics_tracker.get_metrics()

    print(f"\nDone in {elapsed:.1f}s")
    print(f"{'='*40}")
    for k, v in metrics.items():
        print(f"  {k:<12} {v:.6f}")
    print(f"{'='*40}")

    out_path = os.path.join(cfg["output_dir"], "metrics.json")
    with open(out_path, "w") as f:
        json.dump({"variant": variant, "prompt_mode": prompt_mode, "metrics": metrics,
                    "elapsed_s": elapsed}, f, indent=2)
    print(f"Saved to {out_path}")

    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Depth evaluation")
    parser.add_argument("--variant", required=True,
                        choices=["baseline", "pooled", "sam-replace", "sam-concat"])
    parser.add_argument("--prompt-mode", default="multiclass",
                        choices=["empty", "multiclass", "singleclass", "classonly"])
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--base-path", required=True)
    parser.add_argument("--sam-checkpoint", default=None)
    parser.add_argument("--output-dir", default="eval_results")
    return parser.parse_args()


def main():
    args = _parse_args()
    run_eval({
        "variant": args.variant,
        "prompt_mode": args.prompt_mode,
        "config_file": args.config_file,
        "model_file": args.model_file,
        "base_path": args.base_path,
        "sam_checkpoint": args.sam_checkpoint,
        "output_dir": args.output_dir,
    })


if __name__ == "__main__":
    main()
