#!/usr/bin/env python
import argparse
import io
import os
import shutil
import tarfile

import h5py
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

REPO = "sayakpaul/nyu_depth_v2"
TRAIN_SHARDS = [f"data/train-{i:06d}.tar" for i in range(12)]
VAL_SHARDS = [f"data/val-{i:06d}.tar" for i in range(2)]


def export_shard(shard, out_dir, rel_prefix, split_lines, start_idx, limit):
    """Download one shard, write its samples, delete it. Returns new running index."""
    scratch = os.path.join(out_dir, "_shard_tmp")
    os.makedirs(scratch, exist_ok=True)
    path = hf_hub_download(REPO, shard, repo_type="dataset",
                           local_dir=scratch, local_dir_use_symlinks=False)
    idx = start_idx
    with tarfile.open(path) as tar:
        for member in tar:
            if not member.name.endswith(".h5"):
                continue
            if limit is not None and idx >= limit:
                break
            with h5py.File(io.BytesIO(tar.extractfile(member).read()), "r") as h5:
                rgb = np.asarray(h5["rgb"][()])        # (3, H, W) uint8
                depth = np.asarray(h5["depth"][()])    # (H, W) float32 metres
            rgb = np.transpose(rgb, (1, 2, 0))
            depth_mm = np.clip(np.rint(depth * 1000.0), 0, 65535).astype(np.uint16)

            stem = f"{idx:06d}"
            Image.fromarray(rgb).save(os.path.join(out_dir, f"{stem}_rgb.jpg"), quality=95)
            Image.fromarray(depth_mm).save(os.path.join(out_dir, f"{stem}_sync_depth.png"))
            split_lines.append(f"{rel_prefix}/{stem}_rgb.jpg {rel_prefix}/{stem}_sync_depth.png")
            idx += 1
    os.remove(path)
    print(f"  {shard}: wrote up to {idx} ({idx - start_idx} this shard)", flush=True)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("NYU_OUT", "datasets/nyu_hf"),
                    help="Output base_path for the export (holds nyu_{train,test}.txt "
                         "and the train/ + test/ image dirs)")
    ap.add_argument("--train-limit", type=int, default=24231)
    ap.add_argument("--skip-val", action="store_true",
                    help="Skip the val/test export (e.g. when re-running only to grow the train set)")
    args = ap.parse_args()

    train_dir = os.path.join(args.out, "train")
    test_dir = os.path.join(args.out, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Val (test) split: export in full for comparable metrics.
    if args.skip_val:
        print("Skipping val export (--skip-val)", flush=True)
    else:
        test_lines = []
        n = 0
        for shard in VAL_SHARDS:
            n = export_shard(shard, test_dir, "test", test_lines, n, None)
        with open(os.path.join(args.out, "nyu_test.txt"), "w") as f:
            f.write("\n".join(test_lines) + "\n")
        print(f"TEST: {n} samples", flush=True)

    # Train split: capped subset.
    train_lines = []
    n = 0
    for shard in TRAIN_SHARDS:
        if n >= args.train_limit:
            break
        n = export_shard(shard, train_dir, "train", train_lines, n, args.train_limit)
    with open(os.path.join(args.out, "nyu_train.txt"), "w") as f:
        f.write("\n".join(train_lines) + "\n")
    print(f"TRAIN: {n} samples (limit {args.train_limit})", flush=True)

    for d in (os.path.join(test_dir, "_shard_tmp"), os.path.join(train_dir, "_shard_tmp")):
        shutil.rmtree(d, ignore_errors=True)
    print(f"Done. base_path={args.out}", flush=True)


if __name__ == "__main__":
    main()
