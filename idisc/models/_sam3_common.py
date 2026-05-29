"""Shared helpers for the SAM3 image + video pixel encoders."""
from typing import Callable, List, Tuple

import numpy as np
import torch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def coerce_img_size(img_size) -> List[int]:
    if isinstance(img_size, np.ndarray):
        return img_size.tolist()
    return list(img_size)


def coerce_prompt_classes(prompt_classes) -> List[str]:
    classes = list(prompt_classes) if prompt_classes else []
    if not classes:
        raise ValueError("prompt_classes must be non-empty for SAM3 encoders.")
    return classes


def denormalize_imagenet(image: torch.Tensor) -> torch.Tensor:
    """ImageNet-normalised float → uint8 RGB at 0..255."""
    mean = IMAGENET_MEAN.to(image.device, image.dtype)
    std = IMAGENET_STD.to(image.device, image.dtype)
    return ((image * std + mean) * 255.0).clamp(0, 255).to(torch.uint8)


def target_level_sizes(hw: Tuple[int, int], n_levels: int) -> List[Tuple[int, int]]:
    """Stride-4/8/16/... pyramid sizes for the iDisc FPN."""
    h, w = int(hw[0]), int(hw[1])
    return [
        (max(1, h // (2 ** (2 + i))), max(1, w // (2 ** (2 + i))))
        for i in range(n_levels)
    ]


def infer_num_levels(backbone) -> int:
    """FPN level count = neck convs minus the SAM3 `scalp` (discarded top levels)."""
    convs = backbone.vision_backbone.convs
    scalp = int(getattr(backbone, "scalp", 0) or 0)
    return max(1, len(convs) - scalp)


def install_decoder_hook(decoder, sink: Callable[[torch.Tensor], None]):
    """Hook a SAM3 transformer decoder so `sink(hs)` is called with the
    last-layer per-slot tokens (shape (K, 256)) on every fire. Returns the
    handle so the caller can `.remove()` it."""

    def _hook(module, args, output):
        for o in (output if isinstance(output, tuple) else (output,)):
            if isinstance(o, torch.Tensor) and o.dim() == 4 and o.shape[-1] == 256:
                hs = o[-1].detach()
                if hs.dim() == 3:
                    hs = hs.squeeze(1)
                sink(hs)
                return

    return decoder.register_forward_hook(_hook)
