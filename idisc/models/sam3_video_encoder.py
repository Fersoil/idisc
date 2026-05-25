"""Sam3VideoPixelEncoder: drop-in replacement for Sam3PixelEncoder that
wraps SAM3's video inference model instead of the image-only processor.

Key difference from Sam3PixelEncoder:
  - Input is an entire CLIP (T, 3, H, W) instead of a single frame.
  - SAM3 VideoInference runs the detector (backbone + grounding decoder)
    on frame 0 with text prompts, then propagates detections through the
    tracker to frames 1..T-1. Queries for later frames are conditioned on
    the tracker's per-object memory — real temporal context.
  - Output is per-frame FPN + per-frame queries: each iDisc per-frame
    forward receives the same shapes as before but with temporally-aware
    queries.

Important: feature_cache is evicted frame-by-frame during propagation, so
FPN must be captured inside the propagation generator loop.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class Sam3VideoPixelEncoder(nn.Module):
    """Wraps SAM3 video inference (detector + tracker) as an iDisc pixel
    encoder. Processes a T-frame clip in one forward call and returns
    per-frame FPN features + temporally-propagated queries."""

    yields_instance_queries: bool = True
    yields_clip: bool = True          # signals training loop to skip flatten
    is_video_encoder: bool = True     # used by sequence visualizers to branch paths
    embed_dims: List[int]

    def __init__(
        self,
        img_size,
        sam_checkpoint: Optional[str] = None,
        prompt_mode: str = "singleclass",
        prompt_classes: Optional[Sequence[str]] = None,
        freeze_sam3: bool = True,
        load_from_HF: Optional[bool] = None,
        confidence_threshold: float = 0.0,
        max_instances: int = -1,
        **kwargs,
    ):
        super().__init__()

        self.img_size = (
            img_size.tolist() if isinstance(img_size, np.ndarray) else list(img_size)
        )
        self.prompt_mode = prompt_mode
        self.prompt_classes = list(prompt_classes) if prompt_classes else []
        self.confidence_threshold = float(confidence_threshold)
        self._num_queries: Optional[int] = None  # inferred on first hook fire

        if not self.prompt_classes:
            raise ValueError(
                "Sam3VideoPixelEncoder requires non-empty prompt_classes."
            )

        if load_from_HF is None:
            load_from_HF = sam_checkpoint is None

        from sam3.model_builder import build_sam3_video_model
        self.video_model = build_sam3_video_model(
            checkpoint_path=sam_checkpoint if not load_from_HF else None,
            apply_temporal_disambiguation=False,
        ).eval()
        # model_builder hardcodes several score thresholds — override with config values.
        # new_det_thresh: minimum detection score to start tracking a new object (default 0.7)
        # score_threshold_detection: NMS score threshold inside the detector (default 0.5)
        # hotstart_delay: objects suppressed for this many frames after detection (default 15)
        #   — with short clips this hides everything, so we disable it.
        self.video_model.new_det_thresh = self.confidence_threshold
        self.video_model.score_threshold_detection = self.confidence_threshold
        self.video_model.hotstart_delay = 0
        self.video_model.hotstart_unmatch_thresh = 0
        self.video_model.hotstart_dup_thresh = 0
        # Hard cap on simultaneously tracked instances (-1 = unlimited).
        # SAM3's tracker memory scales linearly with this; with threshold=0.0
        # up to 200 instances can be created per clip.
        if max_instances > 0:
            self.video_model.max_num_objects = max_instances

        if freeze_sam3:
            for p in self.video_model.parameters():
                p.requires_grad_(False)
        self._freeze_sam3 = freeze_sam3

        # Infer FPN level count from the detector's vision backbone.
        n_levels = self._infer_num_levels()
        self.embed_dims = [256] * n_levels

        # Per-clip decoder hidden-state buffer: {frame_idx: Tensor(K, 256)}.
        # Populated by the forward hook during propagate_in_video.
        self._decoder_hs_per_frame: Dict[int, torch.Tensor] = {}
        self._decoder_handle = None   # registered lazily (post-deepcopy)

        # Set to True by visualization scripts that need per-frame masklet data.
        # Off by default so training does not pay the CPU cost of converting SAM3
        # numpy mask outputs to tensors for every clip.
        self.track_masklets: bool = False

        # Per-frame tracker masklets. Filled during forward() only when
        # track_masklets=True; one dict per frame:
        #   {"frame_idx": int, "masks": (M, Hm, Wm) float [0,1] or None,
        #    "ids": (M,) long, "scores": (M,) float}
        self._masklets_per_frame: List[Optional[dict]] = []

        # FPN strategy: we call backbone.forward_image(img_batch) ONCE per
        # clip (at the start of forward), cache the dict, and reuse it for
        # both (a) building our per-frame FPN tensors and (b) feeding back
        # into SAM3's internal pipeline so it doesn't re-run the backbone.
        # The reuse is done by patching Sam3Image._get_img_feats to inject
        # our cached dict into the per-frame `backbone_out` dict that SAM3
        # constructs in run_backbone_and_detection. With "backbone_fpn"
        # already present, _get_img_feats takes its fast path and skips
        # its own self.backbone.forward_image fallback.
        #
        # Why this convoluted path instead of just monkey-patching
        # backbone.forward_image globally:
        #   - SAM3's video pipeline routes through forward_grounding which
        #     uses self.backbone.forward_image as a *fallback* inside
        #     _get_img_feats. That fallback is called via Python attribute
        #     lookup, but if SAM3 internally bypasses it (e.g. via direct
        #     vision_backbone.forward(samples) in vl_combiner.py:88), the
        #     patch silently misses — and we have no observable signal
        #     until BN stats show zeros at the end of training.
        #   - Caching via _get_img_feats is the consumer-side intercept:
        #     if SAM3 wants FPN features, it has to go through this method,
        #     so injecting the cached dict guarantees the skip works.
        self._cached_bb_out: Optional[dict] = None
        self._install_get_img_feats_cache()

    def _install_get_img_feats_cache(self):
        """Patch Sam3Image._get_img_feats on the detector instance so it
        injects our cached backbone_out (set in forward() before propagate)
        instead of re-running the backbone."""
        detector = self.video_model.detector
        original = detector._get_img_feats
        enc = self

        def _patched(backbone_out, img_ids):
            if "backbone_fpn" not in backbone_out and enc._cached_bb_out is not None:
                backbone_out.update(enc._cached_bb_out)
            return original(backbone_out, img_ids)

        detector._get_img_feats = _patched

    def _infer_num_levels(self) -> int:
        try:
            convs = self.video_model.detector.backbone.vision_backbone.convs
            scalp = int(
                getattr(self.video_model.detector.backbone, "scalp", 0) or 0
            )
            return max(1, len(convs) - scalp)
        except AttributeError:
            return 3

    def _ensure_decoder_hook(self):
        """Register hook on detector.transformer.decoder lazily so the
        closure captures the live (post-deepcopy) self."""
        if self._decoder_handle is not None:
            return
        decoder = self.video_model.detector.transformer.decoder

        def _hook(module, args, output):
            # clone_output_wrapper does not change shape; output is
            # (num_layers, num_queries, batch=1, 256).
            for o in (output if isinstance(output, tuple) else (output,)):
                if isinstance(o, torch.Tensor) and o.dim() == 4 and o.shape[-1] == 256:
                    last_hs = o[-1].detach()   # (num_queries, 1, 256)
                    if last_hs.dim() == 3:
                        last_hs = last_hs.squeeze(1)  # (num_queries, 256)
                    # Frame index is unknown at hook time; store temporarily
                    # and index by frame in the propagation loop.
                    self._last_hook_hs = last_hs
                    return

        self._decoder_handle = decoder.register_forward_hook(_hook)
        self._last_hook_hs: Optional[torch.Tensor] = None

    def _denormalize(self, image: torch.Tensor) -> torch.Tensor:
        mean = IMAGENET_MEAN.to(image.device, image.dtype)
        std = IMAGENET_STD.to(image.device, image.dtype)
        return ((image * std + mean) * 255.0).clamp(0, 255).to(torch.uint8)

    @staticmethod
    def _target_level_sizes(
        input_hw: Tuple[int, int], n_levels: int
    ) -> List[Tuple[int, int]]:
        h, w = int(input_hw[0]), int(input_hw[1])
        return [
            (max(1, h // (2 ** (2 + i))), max(1, w // (2 ** (2 + i))))
            for i in range(n_levels)
        ]

    def _to_pil_list(self, clip_uint8: torch.Tensor) -> List[Image.Image]:
        """clip_uint8: (T, 3, H, W) uint8 → list of T PIL images."""
        pil = []
        for t in range(clip_uint8.shape[0]):
            arr = clip_uint8[t].permute(1, 2, 0).cpu().numpy()
            pil.append(Image.fromarray(arr))
        return pil

    def forward(self, clip: torch.Tensor):
        """
        Args:
            clip: (T, 3, H, W) ImageNet-normalised fp32.

        Returns:
            (*fpn_levels, queries) where each fpn level has shape
            (T, C, H_stride, W_stride) and queries has shape (T, K, 256).
        """
        if self._freeze_sam3:
            self.video_model.eval()
        self._ensure_decoder_hook()

        T, C, H, W = clip.shape
        clip_uint8 = self._denormalize(clip)
        pil_images = self._to_pil_list(clip_uint8)

        target_sizes = self._target_level_sizes((H, W), len(self.embed_dims))
        n_levels = len(self.embed_dims)

        # Per-frame accumulators: lists of length T.
        per_frame_fpn: List[Optional[List[torch.Tensor]]] = [None] * T
        per_frame_queries: List[Optional[torch.Tensor]] = [None] * T

        self._decoder_hs_per_frame = {}
        self._last_hook_hs = None
        self._raw_backbone_out_buf = None
        self._masklets_per_frame = [None] * T if self.track_masklets else []

        with torch.inference_mode(), torch.autocast(
            device_type=clip.device.type,
            dtype=torch.bfloat16,
            enabled=clip.device.type == "cuda",
        ):
            # init_state preprocesses the PIL images using SAM3's expected
            # normalization (mean=std=(0.5,0.5,0.5)) and resize to image_size
            # (typically 1024). We need to call this BEFORE running the
            # backbone so the input distribution matches what SAM3 was
            # trained on.
            state = self.video_model.init_state(resource_path=pil_images)

            # --- Run backbone ONCE on the preprocessed batch ---
            # We use this output both for our FPN extraction below AND as
            # the cache that the patched _get_img_feats injects into SAM3's
            # internal pipeline (so it skips its own backbone fallback).
            img_batch = state["input_batch"].img_batch
            if not isinstance(img_batch, torch.Tensor):
                img_batch = torch.stack([t for t in img_batch], dim=0)
            bb_out = self.video_model.detector.backbone.forward_image(img_batch)
            self._cached_bb_out = bb_out
            fpn_raw = bb_out.get("backbone_fpn")
            if fpn_raw is None or len(fpn_raw) < n_levels:
                raise RuntimeError(
                    "backbone.forward_image did not return backbone_fpn with "
                    f">= {n_levels} levels (got {len(fpn_raw) if fpn_raw else 0})"
                )
            # Use the highest-resolution n_levels (matches the slicing the
            # rest of iDisc expects via _target_level_sizes).
            for lvl_i, feat_any in enumerate(list(fpn_raw)[:n_levels]):
                # Unwrap NestedTensor → plain tensor if needed.
                feat = getattr(feat_any, "tensors", feat_any).float()
                if feat.dim() == 3:
                    feat = feat.unsqueeze(0)
                tgt = target_sizes[lvl_i]
                if feat.shape[-2:] != tgt:
                    feat = F.interpolate(
                        feat, size=tgt, mode="bilinear", align_corners=False
                    )
                # feat is (T, C, h, w). Distribute one per frame so the
                # per-frame stacking below cleanly yields (T, C, h, w).
                for t in range(T):
                    if per_frame_fpn[t] is None:
                        per_frame_fpn[t] = []
                    per_frame_fpn[t].append(feat[t : t + 1])

            # --- Tracker + decoder hooks for queries / masklets ---
            # Single prompt with all classes combined on frame 0;
            # tracker propagates to frames 1..T-1.
            # All 200 decoder slots are used (no top-K filtering).
            prompt_str = " . ".join(self.prompt_classes)
            self.video_model.add_prompt(state, frame_idx=0, text_str=prompt_str)

            for frame_idx, outputs in self.video_model.propagate_in_video(state):
                # FPN already captured above by the explicit backbone call.
                # Capture queries from the decoder hook (fired during detector
                # forward inside propagate_in_video).
                if self._last_hook_hs is not None:
                    hs = self._last_hook_hs.float()
                    self._num_queries = hs.shape[0]
                    per_frame_queries[frame_idx] = hs
                    self._last_hook_hs = None

                # 3. Capture tracker masklets (only when visualization mode is on).
                if self.track_masklets:
                    raw_masks = outputs.get("out_binary_masks")
                    ids       = outputs.get("out_obj_ids")
                    scores    = outputs.get("out_probs")

                    if raw_masks is not None and raw_masks.shape[0] > 0:
                        masks_f = torch.from_numpy(raw_masks).float()  # (M, H, W)
                        if masks_f.dim() == 4:
                            masks_f = masks_f[:, 0]
                        if masks_f.min() < -0.1 or masks_f.max() > 1.1:
                            masks_f = masks_f.sigmoid()
                        ids_t = (torch.from_numpy(ids).long() if ids is not None
                                 else torch.arange(masks_f.shape[0], dtype=torch.long))
                        scores_t = (torch.from_numpy(scores).float() if scores is not None
                                    else masks_f.flatten(1).amax(1))
                        self._masklets_per_frame[frame_idx] = {
                            "frame_idx": frame_idx,
                            "masks":  masks_f,
                            "ids":    ids_t,
                            "scores": scores_t,
                        }
                    else:
                        self._masklets_per_frame[frame_idx] = {
                            "frame_idx": frame_idx, "masks": None, "ids": None, "scores": None,
                        }

        # Explicitly free the SAM3 inference state (holds all frame features on GPU)
        # and the cached backbone output, then return that memory to the pool
        # before the caller runs the depth head.
        del state
        self._cached_bb_out = None
        torch.cuda.empty_cache()

        # Stack into (T, ...) tensors.
        # Fallback: any frame whose FPN was not captured → zeros (shouldn't happen).
        placeholder_fpn = [
            torch.zeros(1, 256, *target_sizes[i], device=clip.device)
            for i in range(n_levels)
        ]
        num_q = self._num_queries or 200
        placeholder_q = torch.zeros(1, num_q, 256, device=clip.device)

        stacked_fpn = []
        for lvl_i in range(n_levels):
            lvl_frames = []
            for t in range(T):
                f = per_frame_fpn[t]
                lvl_frames.append(f[lvl_i] if f is not None else placeholder_fpn[lvl_i])
            stacked_fpn.append(torch.cat(lvl_frames, dim=0))  # (T, 256, h, w)

        query_frames = []
        for t in range(T):
            q = per_frame_queries[t]
            if q is not None:
                query_frames.append(q.unsqueeze(0))
            else:
                query_frames.append(placeholder_q)
        queries = torch.cat(query_frames, dim=0)  # (T, K, 256)

        return (*stacked_fpn, queries)
