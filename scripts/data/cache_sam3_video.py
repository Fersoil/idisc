#!/usr/bin/env python3
"""
Cache SAM3 video hidden states for KITTI Eigen split sequences.

For each drive sequence:
  1. Run SAM3 video predictor with tracking
  2. Extract last-layer decoder queries (hs[-1]) per frame — all slots
  3. Save queries to disk as .pt files

Usage:
  python scripts/cache_sam3_video.py \
    --manifest splits/kitti/sequence_manifest.json \
    --kitti-root /work/courses/3dv/team17/idisc/datasets/kitti \
    --cache-dir /work/courses/3dv/team17/sam3_cache \
    --checkpoint /work/courses/3dv/team17/sam3_checkpoints/sam3.pt
"""

import argparse
import gc
import json
import os
import time
from typing import Any

import torch
from PIL import Image

KITTI_CLASSES = ["car", "truck", "person", "bicycle", "building", "tree", "road sign", "pole"]

MAX_CHUNK_FRAMES = 100
CHUNK_OVERLAP = 10


def propagate_and_collect(predictor, session_id):
    """Run propagation and return per-frame hidden states."""
    for _ in predictor.handle_stream_request(
        request=dict(type="propagate_in_video", session_id=session_id)
    ):
        pass
    return dict(predictor.hidden_states)


def extract_all_queries(hs_tensor):
    """Extract all queries from hs[-1] (last decoder layer).

    Args:
        hs_tensor: shape (num_layers, batch, num_queries, d_model)

    Returns:
        Tensor of shape (num_queries, d_model) in float16 for storage efficiency.
    """
    last_layer = hs_tensor[-1]  # (batch, num_queries, d_model)
    if last_layer.dim() == 3:
        last_layer = last_layer.squeeze(0)  # (num_queries, d_model)
    return last_layer.half().cpu()


def _run_chunk(predictor, pil_images, local_to_global, frames_to_cache,
               seq_cache_dir):
    """Run SAM3 video predictor on a chunk of PIL images.

    Args:
        pil_images: list of PIL.Image for this chunk
        local_to_global: dict mapping local chunk index -> global frame number
        frames_to_cache: set of global frame indices we need
        seq_cache_dir: output directory

    Returns:
        Number of frames cached from this chunk.
    """
    response = predictor.handle_request(
        request=dict(type="start_session", resource_path=pil_images)
    )
    session_id = response["session_id"]

    for cls_name in KITTI_CLASSES:
        predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=cls_name,
            )
        )

    hidden_states = propagate_and_collect(predictor, session_id)

    cached_count = 0
    for local_idx, hs in hidden_states.items():
        global_idx = local_to_global.get(local_idx)
        if global_idx is not None and global_idx in frames_to_cache:
            out_path = os.path.join(seq_cache_dir, f"{global_idx:010d}.pt")
            if not os.path.exists(out_path):
                queries = extract_all_queries(hs)
                torch.save(queries, out_path)
                cached_count += 1

    predictor.handle_request(
        request=dict(type="close_session", session_id=session_id)
    )
    del hidden_states
    gc.collect()
    torch.cuda.empty_cache()
    return cached_count


def process_sequence(predictor, seq_key, seq_info, kitti_root, cache_dir):
    """Process one drive sequence through SAM3 video predictor, chunked for memory."""
    image_dir = os.path.join(kitti_root, seq_info["image_dir"])
    if not os.path.isdir(image_dir):
        print(f"  SKIP {seq_key}: image dir not found at {image_dir}")
        return 0

    img_names = sorted(
        [f for f in os.listdir(image_dir) if f.endswith(".png")],
        key=lambda p: int(os.path.splitext(p)[0]),
    )
    n_images = len(img_names)
    if n_images == 0:
        print(f"  SKIP {seq_key}: no images found")
        return 0

    frames_to_cache = set(seq_info["frames"])
    seq_cache_dir = os.path.join(cache_dir, seq_info["drive"])
    os.makedirs(seq_cache_dir, exist_ok=True)

    already_cached = set()
    for fn in os.listdir(seq_cache_dir):
        if fn.endswith(".pt"):
            already_cached.add(int(fn.replace(".pt", "")))
    remaining = frames_to_cache - already_cached
    if not remaining:
        print(f"  SKIP {seq_key}: all {len(frames_to_cache)} frames already cached")
        return 0

    n_chunks = max(1, (n_images - CHUNK_OVERLAP) // (MAX_CHUNK_FRAMES - CHUNK_OVERLAP) +
                   (1 if (n_images - CHUNK_OVERLAP) % (MAX_CHUNK_FRAMES - CHUNK_OVERLAP) > 0 else 0))
    print(f"  Processing {seq_key}: {n_images} images, "
          f"{len(remaining)} frames to cache, {n_chunks} chunk(s)...", flush=True)

    t0 = time.time()
    total_cached = 0
    chunk_start = 0

    while chunk_start < n_images:
        chunk_end = min(chunk_start + MAX_CHUNK_FRAMES, n_images)
        chunk_names = img_names[chunk_start:chunk_end]

        pil_images = [
            Image.open(os.path.join(image_dir, name)).convert("RGB")
            for name in chunk_names
        ]

        local_to_global = {
            i: int(os.path.splitext(chunk_names[i])[0])
            for i in range(len(chunk_names))
        }
        g_start = local_to_global[0]
        g_end = local_to_global[len(chunk_names) - 1]

        print(f"    Chunk [{chunk_start}:{chunk_end}] "
              f"(global frames {g_start}-{g_end})...", flush=True)

        n = _run_chunk(predictor, pil_images, local_to_global, frames_to_cache,
                       seq_cache_dir)
        total_cached += n

        # Close PIL images
        for img in pil_images:
            img.close()
        del pil_images

        # Advance past chunk, minus overlap for temporal continuity
        if chunk_end >= n_images:
            break
        chunk_start = chunk_end - CHUNK_OVERLAP

    elapsed = time.time() - t0
    print(f"    Done: cached {total_cached}/{len(remaining)} remaining frames "
          f"in {elapsed:.1f}s ({n_images / max(elapsed, 0.1):.1f} fps)", flush=True)
    return total_cached


def run_cache(cfg: dict[str, Any]) -> dict[str, Any]:
    with open(cfg["manifest"], "r", encoding="utf-8") as f:
        manifest = json.load(f)

    os.makedirs(cfg["cache_dir"], exist_ok=True)

    print("Loading SAM3 video predictor...", flush=True)
    from sam3.model_builder import build_sam3_video_predictor
    predictor = build_sam3_video_predictor(checkpoint_path=cfg.get("checkpoint"))
    print("  SAM3 loaded.", flush=True)

    total_cached = 0
    seq_keys = sorted(manifest.keys())
    start_idx = int(cfg.get("start_idx", 0))
    for i, seq_key in enumerate(seq_keys):
        if i < start_idx:
            continue
        seq_info = manifest[seq_key]
        print(f"\n[{i+1}/{len(seq_keys)}] {seq_key}", flush=True)
        n = process_sequence(
            predictor,
            seq_key,
            seq_info,
            cfg["kitti_root"],
            cfg["cache_dir"],
        )
        total_cached += n

    print(f"\nDone. Total cached: {total_cached} frames")
    print(f"Cache dir: {cfg['cache_dir']}")
    return {"total_cached": int(total_cached), "cache_dir": cfg["cache_dir"]}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache SAM3 video features for KITTI")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to sequence_manifest.json")
    parser.add_argument("--kitti-root", type=str, required=True,
                        help="Root dir of KITTI dataset")
    parser.add_argument("--cache-dir", type=str, required=True,
                        help="Output directory for cached queries")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="SAM3 checkpoint path")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Start from this sequence index (for resuming)")
    return parser.parse_args()


def main():
    args = _parse_args()
    run_cache(
        {
            "manifest": args.manifest,
            "kitti_root": args.kitti_root,
            "cache_dir": args.cache_dir,
            "checkpoint": args.checkpoint,
            "start_idx": args.start_idx,
        }
    )


if __name__ == "__main__":
    main()
