"""Shared helpers for the visualize_* scripts."""
from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.colors import BoundaryNorm, ListedColormap

from idisc.models._sam3_common import (IMAGENET_MEAN, IMAGENET_STD,
                                        letterbox_geometry)


def content_box(orig_hw, H: int, W: int, size: int = 1008):
    """Pixel slice (top, left, h, w) of the real-content band inside an (H, W)
    letterboxed-square display, for a KITTI image of original shape `orig_hw`.
    KITTI is wide, so the square has large top/bottom padding; this is the same
    geometry the model crops its prediction with (`letterbox_geometry`)."""
    ft, fl, fh, fw = letterbox_geometry(orig_hw, size)
    return round(ft * H), round(fl * W), max(1, round(fh * H)), max(1, round(fw * W))


def crop_to_content(arr, orig_hw, size: int = 1008):
    """Crop a letterboxed-square array (H, W[, C]) back to the KITTI content band."""
    t, l, h, w = content_box(orig_hw, arr.shape[0], arr.shape[1], size)
    return arr[t:t + h, l:l + w]

ISD_LABELS = {
    "linear_proj": "SAM3 IDR assignment",
    "replace": "SAM3 IDR assignment",  # legacy alias of linear_proj
}
# "replace" kept for viz of historical run snapshots (renamed to linear_proj).
SAM_MODE_CHOICES = ["linear_proj", "replace"]


def denormalize_to_float01_hwc(img_tensor: torch.Tensor) -> np.ndarray:
    """ImageNet-normalised CHW tensor → HxWx3 float in [0, 1] for imshow."""
    mean = IMAGENET_MEAN.to(img_tensor.device)
    std = IMAGENET_STD.to(img_tensor.device)
    return (img_tensor * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().float().numpy()


def make_writer(fmt: str, fps: int):
    """matplotlib animation writer for gif / mp4."""
    if fmt == "mp4":
        import shutil
        if shutil.which("ffmpeg") is None:
            try:
                import imageio_ffmpeg
                plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                raise RuntimeError(
                    "format=mp4 requires ffmpeg, which was not found. "
                    "Install ffmpeg (or the imageio-ffmpeg package), or use --format gif."
                )
        return FFMpegWriter(fps=fps)
    return PillowWriter(fps=fps)


class AttentionCapture:
    """Collect AFP and ISD cross-attention maps without touching model code."""

    def __init__(self, model) -> None:
        self.afp_attn: Dict[int, torch.Tensor] = {}
        self.afp_hw: Dict[int, Tuple[int, int]] = {}
        self.isd_attn: Dict[int, torch.Tensor] = {}
        self.isd_hw: Dict[int, Tuple[int, int]] = {}
        self._handles: List = []
        self._register(model)

    def _register(self, model) -> None:
        def _afp_pre(module, args):
            for idx, fm in enumerate(args[0]):
                self.afp_hw[idx] = (int(fm.shape[-2]), int(fm.shape[-1]))

        self._handles.append(model.afp.register_forward_pre_hook(_afp_pre))

        # AFP attention: cross_attn_{i+1}_d1 is called `iters` times — the
        # hook overwrites, so we keep the last iteration.
        for i in range(model.afp.num_resolutions):
            ca = getattr(model.afp, f"cross_attn_{i+1}_d1")

            def _make_afp_hook(idx):
                def hook(module, inputs, outputs):
                    self.afp_attn[idx] = inputs[0].detach().cpu()
                return hook

            self._handles.append(ca.dropout.register_forward_hook(_make_afp_hook(i)))

        for i in range(model.isd.num_resolutions):
            head = getattr(model.isd, f"head_{i+1}")

            def _make_isd_pre(idx):
                def hook(module, args):
                    self.isd_hw[idx] = (int(args[0].shape[-2]), int(args[0].shape[-1]))
                return hook

            self._handles.append(head.register_forward_pre_hook(_make_isd_pre(i)))

            last_ca = getattr(head, f"cross_attn_{head.depth}")

            def _make_isd_hook(idx):
                def hook(module, inputs, outputs):
                    self.isd_attn[idx] = inputs[0].detach().cpu()
                return hook

            self._handles.append(last_ca.dropout.register_forward_hook(_make_isd_hook(i)))

    def reset(self) -> None:
        self.afp_attn.clear()
        self.isd_attn.clear()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def extract_sample(
    attn_bh: torch.Tensor, num_heads: int, sample_idx: int
) -> torch.Tensor:
    """(B*H, N, K) → (B, H, N, K), select sample, mean over heads → (N, K)."""
    B = attn_bh.shape[0] // num_heads
    attn = attn_bh.view(B, num_heads, *attn_bh.shape[1:])
    return attn[sample_idx].mean(0).float()


def to_spatial(flat: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """(N, H*W) → (N, H, W)."""
    return flat.reshape(flat.shape[0], h, w)


def isd_assignments(fd: dict, num_heads: int, soft: bool = False) -> Dict[int, np.ndarray]:
    """Per-resolution dominant-IDR maps (or weighted-mean if soft=True).

    soft=True is stable across frames: split attention shows as a blend
    rather than flickering between two IDR indices."""
    out: Dict[int, np.ndarray] = {}
    for res_idx, attn_bh in fd["isd_attn"].items():
        h, w = fd["isd_hw"][res_idx]
        attn = extract_sample(attn_bh, num_heads, sample_idx=0)
        if soft:
            idx = torch.arange(attn.shape[-1], dtype=torch.float32)
            out[res_idx] = (attn * idx).sum(-1).reshape(h, w).numpy()
        else:
            out[res_idx] = attn.argmax(dim=-1).reshape(h, w).numpy()
    return out


def idr_display(num_idrs: int, soft: bool):
    """(cmap, norm_or_None, vmin, vmax) for an ISD assignment panel."""
    if soft:
        return plt.get_cmap("viridis"), None, 0.0, float(num_idrs - 1)
    base = plt.get_cmap("tab20").colors
    colors = [base[i % 20] for i in range(num_idrs)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(num_idrs + 1) - 0.5, num_idrs)
    return cmap, norm, 0, num_idrs - 1


def slot_cmap(n: int):
    """Discrete tab20-based (cmap, norm) for `n` slot indices."""
    cmap, norm, _, _ = idr_display(n, soft=False)
    return cmap, norm


def depth_cmap():
    """plasma with grey for masked / missing-GT pixels."""
    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad(color="#444444")
    return cmap
