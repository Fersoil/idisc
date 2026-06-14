"""SAM3 video encoder used as iDisc pixel_encoder.

Runs the detector on frame 0 with text prompts, then propagates detections
to frames 1..T-1 via the tracker. Returns `(*per_frame_fpn, per_frame_queries)`.
"""
from __future__ import annotations

import contextlib
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from idisc.models._sam3_common import (
    coerce_img_size,
    coerce_prompt_classes,
    denormalize_imagenet,
    infer_num_levels,
    install_decoder_hook,
    target_level_sizes,
)


class Sam3VideoPixelEncoder(nn.Module):
    yields_instance_queries: bool = True
    is_video_encoder: bool = True
    embed_dims: List[int]

    def __init__(
        self,
        img_size,
        sam_checkpoint: Optional[str] = None,
        prompt_classes: Optional[Sequence[str]] = None,
        freeze_sam3: bool = True,
        load_from_HF: Optional[bool] = None,
        confidence_threshold: float = 0.0,
        max_instances: int = -1,
        **kwargs,
    ):
        super().__init__()
        from sam3.model_builder import build_sam3_video_model

        self.img_size = coerce_img_size(img_size)
        self.prompt_classes = coerce_prompt_classes(prompt_classes)

        if load_from_HF is None:
            load_from_HF = sam_checkpoint is None
        self.video_model = build_sam3_video_model(
            checkpoint_path=sam_checkpoint if not load_from_HF else None,
            # Use the plain video tracker (cross-frame memory propagation) and
            # skip the long-video instance-disambiguation heuristics, which
            # suppress objects on the short clips used here.
            apply_temporal_disambiguation=False,
        ).eval()
        # model_builder bakes in a 15-frame hotstart suppression that hides
        # every object on short clips; disable it and override thresholds.
        self.video_model.new_det_thresh = confidence_threshold
        self.video_model.score_threshold_detection = confidence_threshold
        self.video_model.hotstart_delay = 0
        self.video_model.hotstart_unmatch_thresh = 0
        self.video_model.hotstart_dup_thresh = 0
        if max_instances > 0:
            self.video_model.max_num_objects = max_instances

        self._freeze_sam3 = freeze_sam3
        if freeze_sam3:
            self.video_model.requires_grad_(False)

        self.embed_dims = [256] * infer_num_levels(
            self.video_model.detector.backbone
        )
        self._decoder_handle = None
        self._last_hs: Optional[torch.Tensor] = None

        # Visualization opt-in: when True, forward() fills _masklets_per_frame.
        self.track_masklets: bool = False
        self._masklets_per_frame: List[Optional[dict]] = []

        # Inject cached backbone_out via _get_img_feats — the single
        # consumer-side hook every SAM3 FPN path funnels through (patching
        # backbone.forward_image misses the vl_combiner bypass).
        self._cached_bb_out: Optional[dict] = None
        detector = self.video_model.detector
        original_get_img_feats = detector._get_img_feats
        enc = self

        def _patched(backbone_out, img_ids):
            if "backbone_fpn" not in backbone_out and enc._cached_bb_out is not None:
                backbone_out.update(enc._cached_bb_out)
            return original_get_img_feats(backbone_out, img_ids)

        detector._get_img_feats = _patched

    def _ensure_decoder_hook(self):
        # Registered lazily so the closure binds the post-deepcopy self.
        if self._decoder_handle is None:
            self._decoder_handle = install_decoder_hook(
                self.video_model.detector.transformer.decoder,
                lambda hs: setattr(self, "_last_hs", hs),
            )

    def forward(self, clip: torch.Tensor):
        if self._freeze_sam3:
            self.video_model.eval()
        self._ensure_decoder_hook()

        T, _, H, W = clip.shape
        n_levels = len(self.embed_dims)
        target_sizes = target_level_sizes((H, W), n_levels)

        clip_uint8 = denormalize_imagenet(clip)
        pil_images = [
            Image.fromarray(clip_uint8[t].permute(1, 2, 0).cpu().numpy())
            for t in range(T)
        ]

        per_frame_queries: List[Optional[torch.Tensor]] = [None] * T
        self._last_hs = None
        self._masklets_per_frame = [None] * T if self.track_masklets else []

        grad_ctx = (
            torch.inference_mode() if self._freeze_sam3 else contextlib.nullcontext()
        )
        with grad_ctx, torch.autocast(
            device_type=clip.device.type,
            dtype=torch.bfloat16,
            enabled=clip.device.type == "cuda",
        ):
            state = self.video_model.init_state(resource_path=pil_images)

            img_batch = state["input_batch"].img_batch
            if not isinstance(img_batch, torch.Tensor):
                img_batch = torch.stack(list(img_batch), dim=0)
            self._cached_bb_out = (
                self.video_model.detector.backbone.forward_image(img_batch)
            )
            fpn_raw = self._cached_bb_out["backbone_fpn"]
            assert len(fpn_raw) >= n_levels, (
                f"backbone_fpn has {len(fpn_raw)} levels, need {n_levels}"
            )
            fpn_per_level = []
            for lvl_i, feat_any in enumerate(list(fpn_raw)[:n_levels]):
                feat = getattr(feat_any, "tensors", feat_any).float()
                if feat.dim() == 3:
                    feat = feat.unsqueeze(0)
                if feat.shape[-2:] != target_sizes[lvl_i]:
                    feat = F.interpolate(
                        feat, size=target_sizes[lvl_i],
                        mode="bilinear", align_corners=False,
                    )
                fpn_per_level.append(feat)  # (T, C, h, w)

            self.video_model.add_prompt(
                state, frame_idx=0, text_str=" . ".join(self.prompt_classes)
            )
            for frame_idx, outputs in self.video_model.propagate_in_video(state):
                if self._last_hs is not None:
                    per_frame_queries[frame_idx] = self._last_hs.float()
                    self._last_hs = None
                if self.track_masklets:
                    self._masklets_per_frame[frame_idx] = _extract_masklet(
                        frame_idx, outputs
                    )

        del state
        self._cached_bb_out = None
        torch.cuda.empty_cache()

        # Tracked-only frames (no fresh detection) may skip the decoder hook;
        # pad those with zeros so all frames share K.
        nonzero = [q for q in per_frame_queries if q is not None]
        assert nonzero, "decoder hook did not fire on any frame"
        K = nonzero[0].shape[0]
        zeros = torch.zeros(K, 256, device=clip.device)
        queries = torch.stack(
            [q if q is not None else zeros for q in per_frame_queries], dim=0
        )

        return (*fpn_per_level, queries)


def _extract_masklet(frame_idx: int, outputs: dict) -> dict:
    raw_masks = outputs.get("out_binary_masks")
    if raw_masks is None or raw_masks.shape[0] == 0:
        return {"frame_idx": frame_idx, "masks": None, "ids": None, "scores": None}
    masks_f = torch.from_numpy(raw_masks).float()
    if masks_f.dim() == 4:
        masks_f = masks_f[:, 0]
    if masks_f.min() < -0.1 or masks_f.max() > 1.1:
        masks_f = masks_f.sigmoid()
    ids = outputs.get("out_obj_ids")
    scores = outputs.get("out_probs")
    return {
        "frame_idx": frame_idx,
        "masks": masks_f,
        "ids": (torch.from_numpy(ids).long() if ids is not None
                else torch.arange(masks_f.shape[0], dtype=torch.long)),
        "scores": (torch.from_numpy(scores).float() if scores is not None
                   else masks_f.flatten(1).amax(1)),
    }
