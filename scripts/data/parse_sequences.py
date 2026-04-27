#!/usr/bin/env python3
"""Parse KITTI Eigen split files to group frames by drive sequence."""

import json
import os
import re
import sys
from collections import defaultdict


def parse_split(split_path):
    """Parse an Eigen split file and group frames by drive sequence.

    Returns dict: {
        "date/date_drive_XXXX_sync": {
            "date": str,
            "drive": str,
            "frames": [int, ...],        # frame numbers in the split
            "image_dir": str,             # relative path to image_02/data
        }
    }
    """
    sequences = defaultdict(lambda: {"frames": set()})
    with open(split_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            img_path = parts[0]
            # e.g. 2011_09_26/2011_09_26_drive_0057_sync/image_02/data/0000000116.png
            m = re.match(
                r"(20\d{2}_\d{2}_\d{2})/(20\d{2}_\d{2}_\d{2}_drive_\d{4}_sync)/image_02/data/(\d+)\.png",
                img_path,
            )
            if not m:
                continue
            date, drive, frame_str = m.groups()
            seq_key = f"{date}/{drive}"
            sequences[seq_key]["date"] = date
            sequences[seq_key]["drive"] = drive
            sequences[seq_key]["image_dir"] = f"{date}/{drive}/image_02/data"
            sequences[seq_key]["frames"].add(int(frame_str))

    for v in sequences.values():
        v["frames"] = sorted(v["frames"])

    return dict(sequences)


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base, "splits", "kitti", "kitti_eigen_train.txt")
    test_path = os.path.join(base, "splits", "kitti", "kitti_eigen_test.txt")
    out_path = os.path.join(base, "splits", "kitti", "sequence_manifest.json")

    train_seqs = parse_split(train_path)
    test_seqs = parse_split(test_path)

    all_keys = sorted(set(list(train_seqs.keys()) + list(test_seqs.keys())))
    manifest = {}
    for key in all_keys:
        entry = {}
        if key in train_seqs:
            entry = dict(train_seqs[key])
            entry["train_frames"] = train_seqs[key]["frames"]
        if key in test_seqs:
            if not entry:
                entry = dict(test_seqs[key])
            entry["test_frames"] = test_seqs[key]["frames"]
        all_frames = sorted(
            set(entry.get("train_frames", []) + entry.get("test_frames", []))
        )
        entry["frames"] = all_frames
        entry["num_frames"] = len(all_frames)
        manifest[key] = entry

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_frames = sum(v["num_frames"] for v in manifest.values())
    print(f"Parsed {len(manifest)} sequences, {total_frames} total frames")
    for key in sorted(manifest.keys()):
        v = manifest[key]
        print(
            f"  {key}: {v['num_frames']} frames "
            f"(train={len(v.get('train_frames', []))}, "
            f"test={len(v.get('test_frames', []))})"
        )
    print(f"\nManifest saved to {out_path}")


if __name__ == "__main__":
    main()
