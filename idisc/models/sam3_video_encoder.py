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

        # Raw backbone FPN buffer: captured via monkey-patch on
        # detector.backbone.forward_image so we get 256-channel features
        # BEFORE tracker conv_s0/conv_s1 projections reduce the channels.
        self._raw_backbone_out_buf: Optional[dict] = None
        self._patch_backbone_forward_image()

    def _patch_backbone_forward_image(self):
        """Monkey-patch backbone.forward_image so we can capture its raw
        output dict (which includes 256-channel `backbone_fpn`) BEFORE the
        video base applies tracker-specific conv_s0/conv_s1 projections."""
        backbone = self.video_model.detector.backbone
        original = backbone.forward_image

        enc = self   # capture self before deepcopy

        def _patched(image, **kw):
            out = original(image, **kw)
            enc._raw_backbone_out_buf = out
            return out

        backbone.forward_image = _patched

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
            state = self.video_model.init_state(resource_path=pil_images)
            # Single prompt with all classes combined on frame 0;
            # tracker propagates to frames 1..T-1.
            # All 200 decoder slots are used (no top-K filtering).
            prompt_str = " . ".join(self.prompt_classes)
            self.video_model.add_prompt(state, frame_idx=0, text_str=prompt_str)

            for frame_idx, outputs in self.video_model.propagate_in_video(state):
                # 1. Capture FPN from the backbone's raw output (captured via
                #    monkey-patched forward_image before tracker conv projections
                #    reduce channels from 256 to whatever the tracker needs).
                fpn_raw = None
                if self._raw_backbone_out_buf is not None:
                    fpn_raw = self._raw_backbone_out_buf.get("backbone_fpn")
                    self._raw_backbone_out_buf = None

                if fpn_raw is not None and len(fpn_raw) >= n_levels:
                    fpn = [t.float() for t in list(fpn_raw)[:n_levels]]
                    resized = []
                    for lvl_i, feat in enumerate(fpn):
                        if feat.dim() == 3:
                            feat = feat.unsqueeze(0)
                        tgt = target_sizes[lvl_i]
                        if feat.shape[-2:] != tgt:
                            feat = F.interpolate(
                                feat, size=tgt, mode="bilinear", align_corners=False
                            )
                        resized.append(feat)
                    per_frame_fpn[frame_idx] = resized

                # 2. Capture all queries from hook (fired during detector
                #    forward inside propagate_in_video).
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
        # and return that memory to the pool before the caller runs the depth head.
        del state
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
