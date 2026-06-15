"""Frozen SAM3 instance masks as a side-branch for grounding iDisc's AFP clustering.

SAM3 stays frozen and outside the autograd graph: it consumes the un-normalised
(augmented) image and returns per-object mask logits in the iDisc image frame, used by
a grounding loss as a soft co-membership target. GPU-only — SAM3 hardcodes CUDA in its
position encoding. The forward runs under no_grad (not inference_mode, whose tensors
cannot re-enter an autograd-tracked loss) and reads the raw `pred_masks` logits, not the
processor's sigmoided/resized output.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2

from idisc.models._sam3_common import letterbox_geometry, letterbox_to_square


def crop_letterbox_to_frame(
    maps: torch.Tensor, hw: Tuple[int, int], size: int = 1008
) -> torch.Tensor:
    """Crop the content band out of (B, C, H_m, W_m) letterbox-frame maps and resize to
    the original `hw` — the inverse of letterbox_to_square at map resolution."""
    ft, fl, fh, fw = letterbox_geometry(hw, size)
    h_m, w_m = maps.shape[-2:]
    top, left = round(ft * h_m), round(fl * w_m)
    h_c, w_c = max(1, round(fh * h_m)), max(1, round(fw * w_m))
    crop = maps[..., top:top + h_c, left:left + w_c]
    return F.interpolate(crop, size=tuple(hw), mode="bilinear", align_corners=False)


class Sam3MaskModule(nn.Module):
    def __init__(
        self,
        sam_checkpoint: Optional[str],
        prompt_classes: List[str],
        confidence_threshold: float = 0.0,
        load_from_HF: Optional[bool] = None,
        letterbox_size: int = 1008,
    ):
        super().__init__()
        from sam3.model_builder import build_sam3_image_model

        if load_from_HF is None:
            load_from_HF = sam_checkpoint is None
        # SAM3 hardcodes CUDA in its position encoding; build on GPU directly.
        self.sam_model = build_sam3_image_model(
            device="cuda" if torch.cuda.is_available() else "cpu",
            checkpoint_path=sam_checkpoint,
            load_from_HF=load_from_HF,
        ).eval()
        self.sam_model.requires_grad_(False)
        self.prompt = " . ".join(prompt_classes)
        self.confidence_threshold = float(confidence_threshold)
        self.letterbox_size = letterbox_size
        self._proc = None
        self._proc_device = None

    def train(self, mode: bool = True):
        # Keep the frozen SAM3 in eval even when the parent model is in train mode: in
        # train mode forward_grounding takes a target-matching branch (find_target=None
        # -> crash), and SAM3 is never trained anyway.
        super().train(mode)
        self.sam_model.eval()
        return self

    def _ensure_processor(self, device: torch.device):
        if self._proc is not None and self._proc_device == device:
            return
        from sam3.model.sam3_image_processor import Sam3Processor

        self._proc = Sam3Processor(
            self.sam_model, device=str(device),
            confidence_threshold=self.confidence_threshold,
        )
        self._proc_device = device

    @torch.no_grad()
    def _run_one(self, img_uint8: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        device = img_uint8.device
        lb = letterbox_to_square(img_uint8, self.letterbox_size)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            x = self._proc.transform(v2.functional.to_image(lb).to(device)).unsqueeze(0)
            backbone_out = self.sam_model.backbone.forward_image(x)
            backbone_out.update(
                self.sam_model.backbone.forward_text([self.prompt], device=device)
            )
            out = self.sam_model.forward_grounding(
                backbone_out=backbone_out,
                find_input=self._proc.find_stage,
                find_target=None,
                geometric_prompt=self.sam_model._get_dummy_prompt(),
            )
        logits = out["pred_masks"][0].float()           # (Q, h_m, w_m) raw logits
        if logits.dim() == 4:
            logits = logits.squeeze(1)
        return crop_letterbox_to_frame(logits.unsqueeze(0), hw, self.letterbox_size)[0]

    def forward(self, image_raw: torch.Tensor) -> torch.Tensor:
        """image_raw: (B, 3, H, W) uint8 -> (B, Q, H, W) float mask logits."""
        self._ensure_processor(image_raw.device)
        hw = tuple(image_raw.shape[-2:])
        masks = [self._run_one(image_raw[b], hw) for b in range(image_raw.shape[0])]
        return torch.stack(masks, dim=0)
