"""Sam3PixelEncoder: drop-in replacement for ResNet/Swin in iDisc.

A single SAM3 forward pass produces both an N-level FPN (read directly
from `state["backbone_out"]["backbone_fpn"]`) and the per-instance
query embeddings (captured via a forward hook on the SAM3 transformer
decoder and top-K-selected by `state["scores"]`). The encoder returns
an (N+1)-tuple `(f_0, ..., f_{N-1}, instance_queries)`; `IDisc.forward`
peels off the trailing tokens when `yields_instance_queries=True`.

SAM3 internally resizes inputs to its native resolution (~1152²), so
its native FPN levels are too coarse spatially relative to the iDisc
input. Each level is interpolated to (H // 2^(2+i), W // 2^(2+i)) of
the actual image shape to match iDisc's stride-4/8/16 pyramid.

SAM3 weights are frozen by default; the surrounding iDisc modules
(pixel_decoder, AFP, ISD, sam3_proj / sam3_translate) train from
random init.
"""
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


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
        top_k_queries: int = 32,
        confidence_threshold: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        from sam3.model_builder import build_sam3_image_model

        self.img_size = (
            img_size.tolist() if isinstance(img_size, np.ndarray) else list(img_size)
        )
        self.prompt_mode = prompt_mode
        self.prompt_classes = list(prompt_classes) if prompt_classes else []
        self.confidence_threshold = float(confidence_threshold)
        self.top_k_queries: int = int(top_k_queries)

        if self.prompt_mode == "none":
            raise ValueError(
                "Sam3PixelEncoder requires prompt_mode != 'none'; "
                "use 'multiclass', 'singleclass', or 'empty'."
            )
        if self.prompt_mode in ("multiclass", "singleclass") and not self.prompt_classes:
            raise ValueError(
                f"prompt_mode='{self.prompt_mode}' requires non-empty prompt_classes."
            )

        if load_from_HF is None:
            load_from_HF = sam_checkpoint is None

        # SAM3 hardcodes CUDA inside its position encoding; instantiate on
        # GPU directly when available. nn.Module.to() will respect the move
        # later but the initial build must target CUDA.
        sam_build_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam_model = build_sam3_image_model(
            device=sam_build_device,
            checkpoint_path=sam_checkpoint,
            load_from_HF=load_from_HF,
        )
        self.sam_model.eval()
        if freeze_sam3:
            for p in self.sam_model.parameters():
                p.requires_grad_(False)
        self._freeze_sam3 = freeze_sam3

        self._proc = None
        self._proc_device = None
        # Counts SAM3 backbone forward calls so the smoke test can assert
        # single-pass behavior.
        self._call_counter: int = 0

        # FPN level count matches the vision-backbone neck minus the
        # `scalp` levels SAM3 discards at runtime.
        self.embed_dims = [256] * self._infer_num_levels()

        # The SAM3 source build does NOT expose `instance_queries` in the
        # processor state — we hook the transformer decoder directly to
        # capture the last-layer per-slot hidden states.
        # The hook is registered lazily in _ensure_processor so that the
        # closure captures the correct `self` after IDisc.build's deepcopy.
        self._decoder_hs_buf: Optional[torch.Tensor] = None
        self._decoder_handle = None

    def _infer_num_levels(self) -> int:
        convs = self.sam_model.backbone.vision_backbone.convs
        scalp = int(getattr(self.sam_model.backbone, "scalp", 0) or 0)
        return max(1, len(convs) - scalp)

    def _register_decoder_hook(self):
        """Hook the SAM3 transformer decoder to capture per-slot hidden
        states. The decoder returns either a (num_layers, num_queries,
        batch, 256) tensor or a tuple containing one. We keep the last
        layer."""
        decoder = self.sam_model.transformer.decoder

        def _matches(t):
            return isinstance(t, torch.Tensor) and t.dim() == 4 and t.shape[-1] == 256

        def _hook(module, args, output):
            if _matches(output):
                self._decoder_hs_buf = output[-1].detach()
                return
            if isinstance(output, tuple):
                for o in output:
                    if _matches(o):
                        self._decoder_hs_buf = o[-1].detach()
                        return

        return decoder.register_forward_hook(_hook)

    def _ensure_processor(self, device: torch.device):
        if self._proc is None or self._proc_device != device:
            from sam3.model.sam3_image_processor import Sam3Processor

            self._proc = Sam3Processor(
                self.sam_model,
                device=str(device),
                confidence_threshold=self.confidence_threshold,
            )
            self._proc_device = device
            # Register here so the closure captures the live (post-deepcopy)
            # self; registering in __init__ binds to the pre-deepcopy instance.
            if self._decoder_handle is None:
                self._decoder_handle = self._register_decoder_hook()
        return self._proc

    def _build_prompt(self) -> str:
        if self.prompt_mode == "empty":
            return ""
        return " . ".join(self.prompt_classes)

    @staticmethod
    def _target_level_sizes(
        input_hw: Tuple[int, int], n_levels: int
    ) -> List[Tuple[int, int]]:
        """Strides 4, 8, 16, ... at the input resolution."""
        h, w = int(input_hw[0]), int(input_hw[1])
        return [
            (max(1, h // (2 ** (2 + i))), max(1, w // (2 ** (2 + i))))
            for i in range(n_levels)
        ]

    def _denormalize(self, image: torch.Tensor) -> torch.Tensor:
        mean = IMAGENET_MEAN.to(image.device, image.dtype)
        std = IMAGENET_STD.to(image.device, image.dtype)
        return ((image * std + mean) * 255.0).clamp(0, 255).to(torch.uint8)

    def _select_topk_queries(self, state) -> torch.Tensor:
        """Pick top-K decoder hidden states by SAM3's per-slot score.
        With confidence_threshold=0.0, all 200 slots survive filtering
        and `state["scores"]` is aligned with the captured hidden
        states."""
        hs = self._decoder_hs_buf
        assert hs is not None, "decoder hook did not fire during set_text_prompt"
        if hs.dim() == 3:
            hs = hs.squeeze(1)  # (num_queries, 256)

        scores = state["scores"].float().flatten()
        k = min(self.top_k_queries, hs.shape[0])
        _, top_idx = scores.topk(k)
        return hs[top_idx].float()

    def _run_sam3_once(self, raw_image: torch.Tensor):
        """One image → (fpn list, instance_queries tensor)."""
        proc = self._proc
        device_type = raw_image.device.type
        self._decoder_hs_buf = None
        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=device_type == "cuda",
        ):
            state = proc.set_image(raw_image)
            if self.prompt_mode == "singleclass":
                # One SAM3 decoder pass per class; concat all candidate
                # slots, then top-K by combined scores. The decoder hook
                # may occasionally not fire (e.g. when the language path
                # short-circuits); skip those classes silently.
                per_class_hs: List[torch.Tensor] = []
                per_class_scores: List[torch.Tensor] = []
                for cls in self.prompt_classes:
                    self._decoder_hs_buf = None
                    proc.set_text_prompt(prompt=cls, state=state)
                    hs = self._decoder_hs_buf
                    if hs is None:
                        continue
                    if hs.dim() == 3:
                        hs = hs.squeeze(1)
                    per_class_hs.append(hs.float())
                    per_class_scores.append(state["scores"].float().flatten())
                assert per_class_hs, "decoder hook did not fire on any prompt class"
                all_hs = torch.cat(per_class_hs, dim=0)
                all_sc = torch.cat(per_class_scores, dim=0)
                k = min(self.top_k_queries, all_hs.shape[0])
                _, top_idx = all_sc.topk(k)
                instance_queries = all_hs[top_idx]
            else:
                proc.set_text_prompt(prompt=self._build_prompt(), state=state)
                instance_queries = self._select_topk_queries(state)

        fpn_raw = state["backbone_out"]["backbone_fpn"]
        n_levels = len(self.embed_dims)
        assert len(fpn_raw) >= n_levels, (
            f"backbone_fpn has fewer than {n_levels} levels (got {len(fpn_raw)})"
        )
        self._call_counter += 1

        fpn = [t.float() for t in list(fpn_raw)[:n_levels]]
        return fpn, instance_queries.float()

    def forward(self, image: torch.Tensor):
        # image: (B, 3, H, W) ImageNet-normalized fp32
        if self._freeze_sam3:
            self.sam_model.eval()
        device = image.device
        self._ensure_processor(device)
        denorm = self._denormalize(image)

        n_levels = len(self.embed_dims)
        per_level: List[List[torch.Tensor]] = [[] for _ in range(n_levels)]
        per_query: List[torch.Tensor] = []
        for b in range(image.shape[0]):
            fpn, iq = self._run_sam3_once(denorm[b])
            for i in range(n_levels):
                per_level[i].append(fpn[i])
            per_query.append(iq)

        target_sizes = self._target_level_sizes(image.shape[-2:], n_levels)
        feats = []
        for i in range(n_levels):
            stacked = [
                t if t.dim() == 4 else t.unsqueeze(0) for t in per_level[i]
            ]
            level = torch.cat(stacked, dim=0)
            tgt_h, tgt_w = target_sizes[i]
            if level.shape[-2:] != (tgt_h, tgt_w):
                level = F.interpolate(
                    level, size=(tgt_h, tgt_w), mode="bilinear", align_corners=False
                )
            feats.append(level)

        max_k = max(q.shape[0] for q in per_query)
        padded = torch.zeros(
            image.shape[0], max_k, 256, device=device, dtype=torch.float32
        )
        for b, q in enumerate(per_query):
            padded[b, : q.shape[0]] = q

        return (*feats, padded)
