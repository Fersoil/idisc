"""SAM3 image encoder used as iDisc pixel_encoder.

Returns `(*fpn_levels, instance_queries)`; IDisc.forward peels off the
trailing queries when `yields_instance_queries=True`.
"""
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from idisc.models._sam3_common import (
    coerce_img_size,
    coerce_prompt_classes,
    denormalize_imagenet,
    infer_num_levels,
    install_decoder_hook,
    target_level_sizes,
)


class Sam3PixelEncoder(nn.Module):
    yields_instance_queries: bool = True
    embed_dims: List[int]

    def __init__(
        self,
        img_size,
        sam_checkpoint: Optional[str] = None,
        prompt_mode: str = "multiclass",
        prompt_classes: Optional[Sequence[str]] = None,
        freeze_sam3: bool = True,
        load_from_HF: Optional[bool] = None,
        confidence_threshold: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        from sam3.model_builder import build_sam3_image_model

        self.img_size = coerce_img_size(img_size)
        self.prompt_mode = prompt_mode
        self.prompt_classes = coerce_prompt_classes(prompt_classes)
        self.confidence_threshold = float(confidence_threshold)

        if load_from_HF is None:
            load_from_HF = sam_checkpoint is None
        # SAM3 hardcodes CUDA in its position encoding; build on GPU directly.
        self.sam_model = build_sam3_image_model(
            device="cuda" if torch.cuda.is_available() else "cpu",
            checkpoint_path=sam_checkpoint,
            load_from_HF=load_from_HF,
        ).eval()
        self._freeze_sam3 = freeze_sam3
        if freeze_sam3:
            self.sam_model.requires_grad_(False)

        self.embed_dims = [256] * infer_num_levels(self.sam_model.backbone)

        # SAM3 doesn't expose decoder hidden states via the processor API;
        # hook the transformer decoder to capture per-slot tokens. Processor
        # and hook are registered lazily so the closure binds the post-deepcopy
        # self (IDisc.build deepcopies the assembled model).
        self._proc = None
        self._proc_device: Optional[torch.device] = None
        self._last_hs: Optional[torch.Tensor] = None
        self._decoder_handle = None

    def _ensure_processor(self, device: torch.device):
        if self._proc is not None and self._proc_device == device:
            return
        from sam3.model.sam3_image_processor import Sam3Processor

        self._proc = Sam3Processor(
            self.sam_model,
            device=str(device),
            confidence_threshold=self.confidence_threshold,
        )
        self._proc_device = device
        if self._decoder_handle is None:
            self._decoder_handle = install_decoder_hook(
                self.sam_model.transformer.decoder,
                lambda hs: setattr(self, "_last_hs", hs),
            )

    def _run_once(self, raw_image: torch.Tensor):
        device_type = raw_image.device.type
        self._last_hs = None
        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=device_type == "cuda",
        ):
            state = self._proc.set_image(raw_image)
            if self.prompt_mode == "singleclass":
                # The language path can short-circuit and skip the hook on
                # some prompts; drop those silently.
                per_class = []
                for cls in self.prompt_classes:
                    self._last_hs = None
                    self._proc.set_text_prompt(prompt=cls, state=state)
                    if self._last_hs is not None:
                        per_class.append(self._last_hs.float())
                assert per_class, "decoder hook did not fire on any prompt class"
                queries = torch.cat(per_class, dim=0)
            else:
                self._proc.set_text_prompt(
                    prompt=" . ".join(self.prompt_classes), state=state,
                )
                assert self._last_hs is not None, "decoder hook did not fire"
                queries = self._last_hs.float()

        fpn_raw = state["backbone_out"]["backbone_fpn"]
        n_levels = len(self.embed_dims)
        assert len(fpn_raw) >= n_levels
        return [t.float() for t in list(fpn_raw)[:n_levels]], queries

    def forward(self, image: torch.Tensor):
        if self._freeze_sam3:
            self.sam_model.eval()
        device = image.device
        self._ensure_processor(device)
        denorm = denormalize_imagenet(image)

        B = image.shape[0]
        n_levels = len(self.embed_dims)
        target_sizes = target_level_sizes(image.shape[-2:], n_levels)

        per_level: List[List[torch.Tensor]] = [[] for _ in range(n_levels)]
        per_query: List[torch.Tensor] = []
        for b in range(B):
            fpn, queries = self._run_once(denorm[b])
            for i in range(n_levels):
                per_level[i].append(fpn[i])
            per_query.append(queries)

        feats = []
        for i in range(n_levels):
            stacked = torch.cat(
                [t if t.dim() == 4 else t.unsqueeze(0) for t in per_level[i]], dim=0,
            )
            if stacked.shape[-2:] != target_sizes[i]:
                stacked = F.interpolate(
                    stacked, size=target_sizes[i], mode="bilinear", align_corners=False,
                )
            feats.append(stacked)

        max_k = max(q.shape[0] for q in per_query)
        padded = torch.zeros(B, max_k, 256, device=device, dtype=torch.float32)
        for b, q in enumerate(per_query):
            padded[b, : q.shape[0]] = q

        return (*feats, padded)
