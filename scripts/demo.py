#!/usr/bin/env python
"""Predict a depth map for a single image with a trained checkpoint.

    python scripts/demo.py --config <resolved_config.yaml> \
        --checkpoint <ckpt.pt> --image <rgb.png> --out depth.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.experiments._eval_common import build_eval_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def letterbox(img, size):
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = round(w * scale), round(h * scale)
    canvas = Image.new("RGB", (size, size))
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (px, py))
    return canvas, (px, py, nw, nh)


def main():
    ap = argparse.ArgumentParser(description="Single-image depth prediction")
    ap.add_argument("--config", required=True, help="resolved_config.yaml of the checkpoint")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="depth.png")
    ap.add_argument("--size", type=int, default=1008,
                    help="square input size, letterboxed (1008 for SAM3, training default)")
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_eval_model(cfg, args.checkpoint, device)
    sam_mode = cfg["method"]["sam_mode"]

    img = Image.open(args.image).convert("RGB")
    boxed, (px, py, nw, nh) = letterbox(img, args.size)
    x = TF.normalize(TF.to_tensor(boxed), IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0).to(device)

    with torch.no_grad():
        pred, _, _ = model(x, sam_mode=sam_mode)

    depth = pred[0, 0].float().cpu().numpy()[py:py + nh, px:px + nw]
    depth = np.asarray(Image.fromarray(depth).resize(img.size, Image.BILINEAR))

    norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    colored = (matplotlib.colormaps["magma"](norm)[..., :3] * 255).astype(np.uint8)
    Image.fromarray(colored).save(args.out)
    print(f"wrote {args.out}  (depth {depth.min():.2f}-{depth.max():.2f} m)", flush=True)


if __name__ == "__main__":
    main()
