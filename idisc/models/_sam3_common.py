from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

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
    mean = IMAGENET_MEAN.to(image.device, image.dtype)
    std = IMAGENET_STD.to(image.device, image.dtype)
    return ((image * std + mean) * 255.0).clamp(0, 255).to(torch.uint8)


def target_level_sizes(hw: Tuple[int, int], n_levels: int) -> List[Tuple[int, int]]:
    h, w = int(hw[0]), int(hw[1])
    return [
        (max(1, h // (2 ** (2 + i))), max(1, w // (2 ** (2 + i))))
        for i in range(n_levels)
    ]


def letterbox_to_square(
    img: torch.Tensor, size: int = 1008, pad_value: int = 128
) -> torch.Tensor:
    c, h, w = img.shape
    scale = size / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    resized = F.interpolate(
        img.unsqueeze(0).float(), size=(new_h, new_w),
        mode="bilinear", align_corners=False,
    ).squeeze(0)
    out = torch.full((c, size, size), float(pad_value),
                     device=img.device, dtype=torch.float32)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    out[:, top:top + new_h, left:left + new_w] = resized
    return out.clamp(0, 255).to(torch.uint8)


def letterbox_geometry(
    hw: Tuple[int, int], size: int = 1008
) -> Tuple[float, float, float, float]:
    h, w = int(hw[0]), int(hw[1])
    scale = size / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    top, left = (size - new_h) // 2, (size - new_w) // 2
    return (top / size, left / size, new_h / size, new_w / size)


def infer_num_levels(backbone) -> int:
    convs = backbone.vision_backbone.convs
    scalp = int(getattr(backbone, "scalp", 0) or 0)
    return max(1, len(convs) - scalp)


def install_decoder_hook(decoder, sink: Callable[[torch.Tensor], None]):

    def _hook(module, args, output):
        for o in (output if isinstance(output, tuple) else (output,)):
            if isinstance(o, torch.Tensor) and o.dim() == 4 and o.shape[-1] == 256:
                hs = o[-1].detach()
                if hs.dim() == 3:
                    hs = hs.squeeze(1)
                sink(hs)
                return

    return decoder.register_forward_hook(_hook)


def install_encoder_hook(encoder, sink: Callable[[dict], None]):

    def _hook(module, args, output):
        if isinstance(output, dict) and "memory" in output:
            sink(output)

    return encoder.register_forward_hook(_hook)
