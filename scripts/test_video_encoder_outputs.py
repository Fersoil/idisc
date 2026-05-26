#!/usr/bin/env python
"""
Diagnostic script: compare SAM3 video encoder outputs.

Tests:
  1. Raw encoder output diagnostics (FPN, queries, masklets)
  2. SAM3 video model directly (bypass our wrapper) — do masklets appear?
  3. Encoder with vs without the _get_img_feats cache patch
  4. Shape inspection at the patch injection point
  5. Depth prediction consistency (pre-extracted vs fresh forward)

Usage:
  python scripts/test_video_encoder_outputs.py \
    --config-file configs/kitti/kitti_sam3_video.json \
    --model-file <checkpoint.pt> \
    --base-path <base_path> \
    --clip-idx 50
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.cuda as tcuda
from PIL import Image

from idisc.dataloders.kitti_sequence import KITTISequenceDataset
from idisc.models.idisc import IDisc

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denorm_to_pil(img_tensor):
    mean = IMAGENET_MEAN.to(img_tensor.device)
    std  = IMAGENET_STD.to(img_tensor.device)
    x = (img_tensor * std + mean).clamp(0, 1)
    arr = (x.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def load_clip(config, base_path, clip_idx, clip_length):
    data_path = os.path.join(base_path, config["data"]["data_root"])
    manifest = os.path.join(base_path,
                            config["data"].get("manifest_path",
                                               "splits/kitti/sequence_manifest.json"))
    ds = KITTISequenceDataset(
        test_mode=True,
        base_path=data_path,
        manifest_path=manifest,
        clip_length=clip_length,
        crop=config["data"]["crop"],
    )
    print(f"Dataset: {len(ds)} clips")
    clip = ds[clip_idx]
    print(f"Clip {clip_idx}: seq={clip['sequence_id']}  frames={clip['frame_indices']}")
    return clip


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# ── Step 1: Raw encoder output diagnostics ──────────────────────────────────

def step1_encoder_diagnostics(model, images, device):
    print_section("Step 1: Raw encoder output diagnostics")

    enc = model.pixel_encoder
    enc.track_masklets = True
    enc.video_model.masklet_confirmation_consecutive_det_thresh = 1

    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda",
    ):
        enc_out = enc(images)

    fpn_levels = enc_out[:-1]
    queries = enc_out[-1]

    print("FPN levels:")
    for i, lvl in enumerate(fpn_levels):
        print(f"  level {i}: shape={tuple(lvl.shape)}  "
              f"min={lvl.min():.4f}  max={lvl.max():.4f}  mean={lvl.mean():.4f}  "
              f"nonzero={(lvl != 0).float().mean():.2%}")

    print(f"\nQueries: shape={tuple(queries.shape)}")
    T = queries.shape[0]
    for t in range(T):
        q = queries[t]
        norm = q.norm(dim=-1)
        print(f"  frame {t}: norm min={norm.min():.4f}  max={norm.max():.4f}  "
              f"mean={norm.mean():.4f}  all_zero={q.abs().max() < 1e-6}")

    print(f"\nMasklets per frame ({len(enc._masklets_per_frame)} entries):")
    for t, m in enumerate(enc._masklets_per_frame):
        if m is None:
            print(f"  frame {t}: None (not populated)")
        elif m["masks"] is None:
            print(f"  frame {t}: masks=None, ids=None, scores=None (empty)")
        else:
            print(f"  frame {t}: masks={tuple(m['masks'].shape)}  "
                  f"ids={m['ids'].tolist()[:10]}{'...' if len(m['ids']) > 10 else ''}  "
                  f"scores min={m['scores'].min():.3f} max={m['scores'].max():.3f}")

    return fpn_levels, queries


# ── Step 2: SAM3 video model directly (bypass wrapper) ─────────────────────

def step2_sam3_direct(images, device, sam_checkpoint, prompt_classes):
    print_section("Step 2: SAM3 video model directly (no wrapper)")

    from sam3.model_builder import build_sam3_video_model
    video_model = build_sam3_video_model(
        checkpoint_path=sam_checkpoint,
        apply_temporal_disambiguation=False,
    ).eval()
    video_model.masklet_confirmation_consecutive_det_thresh = 1
    video_model.hotstart_delay = 0
    video_model.hotstart_unmatch_thresh = 0
    video_model.hotstart_dup_thresh = 0
    video_model.new_det_thresh = 0.3
    video_model.score_threshold_detection = 0.3
    video_model.max_num_objects = 15

    T = images.shape[0]
    pil_images = []
    for t in range(T):
        mean = IMAGENET_MEAN.to(images.device)
        std  = IMAGENET_STD.to(images.device)
        x = ((images[t] * std + mean) * 255).clamp(0, 255).to(torch.uint8)
        arr = x.permute(1, 2, 0).cpu().numpy()
        pil_images.append(Image.fromarray(arr))

    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda",
    ):
        state = video_model.init_state(resource_path=pil_images)
        prompt_str = " . ".join(prompt_classes)
        video_model.add_prompt(state, frame_idx=0, text_str=prompt_str)

        print(f"Prompt: '{prompt_str}'")
        for frame_idx, outputs in video_model.propagate_in_video(state):
            masks = outputs.get("out_binary_masks")
            ids = outputs.get("out_obj_ids")
            scores = outputs.get("out_probs")
            if masks is not None and len(masks) > 0:
                print(f"  frame {frame_idx}: {masks.shape[0]} masklets  "
                      f"ids={ids.tolist()[:10]}  "
                      f"scores min={scores.min():.3f} max={scores.max():.3f}")
            else:
                print(f"  frame {frame_idx}: 0 masklets")

    del video_model, state
    torch.cuda.empty_cache()


# ── Step 3: With vs without _get_img_feats patch ───────────────────────────

def step3_patch_comparison(model, images, device):
    print_section("Step 3: With vs without _get_img_feats cache patch")

    enc = model.pixel_encoder
    enc.track_masklets = True
    enc.video_model.masklet_confirmation_consecutive_det_thresh = 1

    # Test A: current code (patch active)
    print("Test A: WITH _get_img_feats cache patch (current code)")
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda",
    ):
        enc_out_a = enc(images)
    masklets_a = list(enc._masklets_per_frame)
    for t, m in enumerate(masklets_a):
        has = m is not None and m["masks"] is not None
        n = m["masks"].shape[0] if has else 0
        print(f"  frame {t}: {n} masklets")
    del enc_out_a
    torch.cuda.empty_cache()

    # Test B: disable the cache injection by clearing _cached_bb_out
    # We monkey-patch the _install method to do nothing for this test
    print("\nTest B: WITHOUT cache patch (force _get_img_feats fallback path)")
    original_install = enc._install_get_img_feats_cache

    # Save and replace the patched _get_img_feats with original
    detector = enc.video_model.detector
    if hasattr(detector, '_original_get_img_feats'):
        original_get_img_feats = detector._original_get_img_feats
    else:
        print("  WARNING: cannot find original _get_img_feats, skipping test B")
        return

    # Instead, we can just ensure _cached_bb_out is never set
    # by wrapping forward to clear it before propagation
    old_cached = enc._cached_bb_out

    # Patch: after backbone call, clear the cache so injection never triggers
    real_forward_image = enc.video_model.detector.backbone.forward_image
    captured_bb_outs = []

    def capturing_forward_image(image, **kw):
        out = real_forward_image(image, **kw)
        captured_bb_outs.append({
            "backbone_fpn_0_shape": out["backbone_fpn"][0].shape if "backbone_fpn" in out else None,
            "sam2_backbone_out": "present" if "sam2_backbone_out" in out else "absent",
        })
        return out

    enc.video_model.detector.backbone.forward_image = capturing_forward_image

    # Neutralize the cache: make it so the patch condition never fires
    class NullCache:
        def __init__(self):
            enc._cached_bb_out = None

    # We need to prevent the explicit backbone call from setting _cached_bb_out
    # The simplest way: temporarily replace forward to skip the explicit call
    # Actually, let's just set _cached_bb_out = None right before propagation
    # by hooking into propagate_in_video

    # Simpler approach: just clear _cached_bb_out after it's set
    original_forward = enc.forward

    def patched_forward(clip):
        result = original_forward(clip)
        return result

    # Actually the simplest: just neutralize the injection
    # Set _cached_bb_out to an empty dict without backbone_fpn
    # so the condition `"backbone_fpn" not in backbone_out` is true
    # but `enc._cached_bb_out is not None` leads to update with empty dict
    # ... this is getting complicated. Let's take a different approach.

    # Just run SAM3 directly (step 2 already does this) and compare.
    print("  (Skipping — Step 2 already tests SAM3 directly without the wrapper)")
    enc.video_model.detector.backbone.forward_image = real_forward_image
    enc._cached_bb_out = old_cached


# ── Step 4: Shape inspection at patch injection point ───────────────────────

def step4_shape_inspection(model, images, device):
    print_section("Step 4: Shape inspection at _get_img_feats injection point")

    enc = model.pixel_encoder
    enc.track_masklets = True
    enc.video_model.masklet_confirmation_consecutive_det_thresh = 1
    detector = enc.video_model.detector

    call_log = []

    # Temporarily replace the patched _get_img_feats with a logging version
    current_get_img_feats = detector._get_img_feats

    def logging_get_img_feats(backbone_out, img_ids):
        entry = {
            "img_ids": img_ids.tolist() if hasattr(img_ids, 'tolist') else str(img_ids),
            "has_backbone_fpn_before": "backbone_fpn" in backbone_out,
        }

        result = current_get_img_feats(backbone_out, img_ids)

        entry["has_backbone_fpn_after"] = "backbone_fpn" in backbone_out
        if "backbone_fpn" in backbone_out:
            entry["backbone_fpn_shapes"] = [tuple(x.shape) for x in backbone_out["backbone_fpn"][:3]]
        if "sam2_backbone_out" in backbone_out:
            sam2 = backbone_out["sam2_backbone_out"]
            if sam2 is not None and "backbone_fpn" in sam2:
                entry["sam2_fpn_shapes"] = [tuple(x.shape) for x in sam2["backbone_fpn"][:3]]
            else:
                entry["sam2_fpn_shapes"] = "None or missing"
        else:
            entry["sam2_fpn_shapes"] = "key absent"
        entry["id_mapping"] = ("present" if backbone_out.get("id_mapping") is not None
                               else "absent")
        call_log.append(entry)
        return result

    detector._get_img_feats = logging_get_img_feats

    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda",
    ):
        enc_out = enc(images)

    detector._get_img_feats = current_get_img_feats

    print(f"_get_img_feats called {len(call_log)} times:\n")
    for i, entry in enumerate(call_log):
        print(f"  Call {i}:")
        print(f"    img_ids: {entry['img_ids']}")
        print(f"    backbone_fpn before injection: {entry['has_backbone_fpn_before']}")
        print(f"    backbone_fpn after injection:  {entry['has_backbone_fpn_after']}")
        if "backbone_fpn_shapes" in entry:
            print(f"    backbone_fpn shapes: {entry['backbone_fpn_shapes']}")
        print(f"    sam2_backbone_out fpn shapes: {entry['sam2_fpn_shapes']}")
        print(f"    id_mapping: {entry['id_mapping']}")

    del enc_out
    torch.cuda.empty_cache()


# ── Step 5: Depth prediction consistency ────────────────────────────────────

def step5_depth_consistency(model, images, depths, masks, device, args_ref=None):
    print_section("Step 5: Depth — sam_mode ablation + FPN order check")

    enc = model.pixel_encoder
    use_amp = device.type == "cuda"
    context = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp)
    t = 0
    gt = depths[t:t+1]
    msk = masks[t:t+1]
    model.eval()

    with torch.no_grad():
        enc_out = enc(images)
        fpn_levels = enc_out[:-1]
        queries_T = enc_out[-1]
        del enc_out
        torch.cuda.empty_cache()

    frame_fpn = tuple(lvl[t:t+1] for lvl in fpn_levels)
    iq = queries_T[t]

    # Test A: sam_mode="replace" with reversed FPN (current approach)
    # Test B: sam_mode="none" (AFP, no queries — isolates FPN quality)
    # Test C: sam_mode="replace" with NON-reversed FPN (check if order is wrong)
    tests = [
        ("replace, reversed FPN",  "replace", tuple(reversed(frame_fpn)), iq),
        ("none (AFP only)",        "none",    tuple(reversed(frame_fpn)), None),
        ("replace, ORIGINAL order","replace", frame_fpn,                  iq),
        ("none, ORIGINAL order",   "none",    frame_fpn,                  None),
    ]

    for label, sam_mode, fpn, queries in tests:
        try:
            with torch.no_grad(), context:
                pred, _, _ = model(
                    images[t:t+1],
                    instance_queries=queries,
                    sam_mode=sam_mode,
                    pre_extracted_encoder_outputs=fpn,
                    gt=gt, mask=msk,
                )

            valid = msk > 0
            abs_rel = "N/A"
            if valid.any():
                abs_rel = f"{((pred[valid] - gt[valid]).abs() / gt[valid].clamp(min=1e-3)).mean():.4f}"

            print(f"  {label}:")
            print(f"    depth: min={pred.min():.2f}  max={pred.max():.2f}  mean={pred.mean():.2f}")
            print(f"    abs_rel: {abs_rel}")
        except Exception as e:
            print(f"  {label}: ERROR — {e}")

    # Also print FPN shapes to confirm order
    print(f"\n  FPN shapes (encoder order): {[tuple(l.shape) for l in frame_fpn]}")
    print(f"  FPN shapes (reversed):      {[tuple(l.shape) for l in reversed(frame_fpn)]}")

    # Test E: run the EXACT validation function from finetune.py
    print("\n  Running finetune's validate_model_sequential on full val set (first 10 batches):")
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "experiments"))
    from finetune import validate_model_sequential
    import idisc.dataloders as custom_dataset
    from torch.utils.data import DataLoader, SequentialSampler
    import json

    with open(args_ref["config_file"]) as f:
        val_config = json.load(f)
    val_config["data"]["val_dataset"] = "KITTISequenceDataset"
    data_path = os.path.join(args_ref["base_path"], val_config["data"]["data_root"])
    valid_dataset = getattr(custom_dataset, val_config["data"]["val_dataset"])(
        test_mode=True, base_path=data_path, crop=val_config["data"]["crop"],
        manifest_path=os.path.join(args_ref["base_path"],
                                   val_config["data"].get("manifest_path", "splits/kitti/sequence_manifest.json")),
        clip_length=val_config["data"].get("clip_length", 4),
    )
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False,
                              num_workers=0, sampler=SequentialSampler(valid_dataset),
                              pin_memory=True, drop_last=False)
    print(f"    Val dataset: {len(valid_dataset)} clips")

    # Limit to 10 batches for speed
    class LimitedLoader:
        def __init__(self, loader, limit):
            self.loader = loader
            self.limit = limit
        def __iter__(self):
            for i, batch in enumerate(self.loader):
                if i >= self.limit:
                    break
                yield batch
        def __len__(self):
            return min(self.limit, len(self.loader))

    model.eval()
    with torch.no_grad():
        metrics = validate_model_sequential(
            model, LimitedLoader(valid_loader, 10), context, device, sam_mode="replace"
        )
    print(f"    Results (10 batches): {metrics}")

    del fpn_levels, queries_T
    torch.cuda.empty_cache()


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Diagnose SAM3 video encoder outputs")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--base-path", required=True)
    parser.add_argument("--clip-idx", type=int, default=50)
    parser.add_argument("--clip-length", type=int, default=4)
    parser.add_argument("--steps", default="1,2,4,5",
                        help="Comma-separated step numbers to run (default: 1,2,4,5)")
    args = parser.parse_args()

    steps = set(int(s) for s in args.steps.split(","))

    with open(args.config_file) as f:
        config = json.load(f)

    device = torch.device("cuda") if tcuda.is_available() else torch.device("cpu")

    clip = load_clip(config, args.base_path, args.clip_idx, args.clip_length)
    images = clip["images"].to(device)
    depths = clip["depths"].to(device)
    masks_t = clip["masks"].to(device)

    sam_checkpoint = config["model"]["pixel_encoder"]["sam_checkpoint"]
    prompt_classes = config["model"]["pixel_encoder"].get("prompt_classes",
                                                          ["vehicle", "tree", "road", "building"])

    if 2 in steps:
        step2_sam3_direct(images, device, sam_checkpoint, prompt_classes)
        torch.cuda.empty_cache()

    model = IDisc.build(config)

    # Check checkpoint key matching before load
    from copy import deepcopy
    ckpt = torch.load(args.model_file, map_location=device)
    new_sd = deepcopy({k.replace("module.", ""): v for k, v in ckpt.items()})
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    non_pe_missing = [k for k in missing if not k.startswith("pixel_encoder")]
    print(f"\nCheckpoint load: {len(new_sd)} keys, "
          f"missing={len(missing)} (non-pixel_encoder={len(non_pe_missing)}), "
          f"unexpected={len(unexpected)}")
    if non_pe_missing:
        print(f"  CRITICAL missing (non-pixel_encoder):")
        for k in sorted(non_pe_missing):
            print(f"    {k}")
    if unexpected:
        print(f"  Unexpected: {sorted(unexpected)[:5]}")
    del ckpt, new_sd

    model = model.to(device).eval()
    print(f"Model loaded: pixel_encoder={type(model.pixel_encoder).__name__}")

    if 1 in steps:
        step1_encoder_diagnostics(model, images, device)
        torch.cuda.empty_cache()

    if 4 in steps:
        step4_shape_inspection(model, images, device)
        torch.cuda.empty_cache()

    if 5 in steps:
        step5_depth_consistency(model, images, depths, masks_t, device,
                                args_ref={"config_file": args.config_file,
                                          "base_path": args.base_path})
        torch.cuda.empty_cache()

    print_section("Done")


if __name__ == "__main__":
    main()
