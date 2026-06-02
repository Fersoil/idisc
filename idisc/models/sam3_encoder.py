"""SAM3 image encoder used as iDisc pixel_encoder.

Returns `(*fpn_levels, instance_queries)`; IDisc.forward peels off the
trailing queries when `yields_instance_queries=True`.
"""
from typing import List, Optional, Sequence

import torch
from torch import nn
from torchvision.transforms import v2

from idisc.models._sam3_common import (
    coerce_img_size,
    coerce_prompt_classes,
    denormalize_imagenet,
    infer_num_levels,
    install_decoder_hook,
    install_encoder_hook,
    letterbox_geometry,
    letterbox_to_square,
)
from idisc.models.id_module import MemoryFPN
from idisc.models.lora import LoRALinear, inject_lora


def _patch_addmm_act_for_grad():
    """SAM3's fused ``addmm_act`` raises when grad is enabled (it detaches weights
    for an inference fast-path). When we run the trunk with grad for LoRA, fall
    back to the plain ``act(linear(x))`` it stands in for. Idempotent; the frozen
    path (grad disabled) still hits the original fused op."""
    import sam3.model.vitdet as vitdet

    if getattr(vitdet.addmm_act, "_grad_aware", False):
        return
    orig = vitdet.addmm_act

    def addmm_act(activation, linear, x):
        if torch.is_grad_enabled():
            return activation()(linear(x))
        return orig(activation, linear, x)

    addmm_act._grad_aware = True
    vitdet.addmm_act = addmm_act


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
        pixel_source: str = "msda",
        sam3_trainable: Optional[Sequence[str]] = None,
        lora: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        from sam3.model_builder import build_sam3_image_model

        self.img_size = coerce_img_size(img_size)
        self.prompt_mode = prompt_mode
        self.prompt_classes = coerce_prompt_classes(prompt_classes)
        self.confidence_threshold = float(confidence_threshold)
        self.pixel_source = pixel_source
        self.letterbox_size = 1008

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

        self.sam3_trainable = list(sam3_trainable or [])
        self._grad_grounding = len(self.sam3_trainable) > 0
        for part in self.sam3_trainable:
            self._sam3_submodule(part).requires_grad_(True)

        # LoRA on the otherwise-frozen ViT trunk. 
        lora = lora or {}
        self._lora = bool(lora.get("enabled", False))
        if self._lora:
            _patch_addmm_act_for_grad()
            trunk = self.sam_model.backbone.vision_backbone.trunk
            n = inject_lora(
                trunk,
                targets=tuple(lora.get("targets", ["qkv", "proj"])),
                r=int(lora.get("rank", 8)),
                alpha=int(lora.get("alpha", 16)),
                dropout=float(lora.get("dropout", 0.0)),
            )
            for m in trunk.modules():
                if isinstance(m, LoRALinear):
                    m.lora_A.requires_grad_(True)
                    m.lora_B.requires_grad_(True)
            self._grad_grounding = True
            print(f"  LoRA on ViT trunk: {n} layers "
                  f"({lora.get('targets', ['qkv', 'proj'])}), "
                  f"r={lora.get('rank', 8)} alpha={lora.get('alpha', 16)}", flush=True)

        self.embed_dims = [256] * infer_num_levels(self.sam_model.backbone)
        self.memory_fpn = (
            MemoryFPN(dim=256, n_levels=len(self.embed_dims))
            if self.pixel_source == "sam3_memory" else None
        )

        self._proc = None
        self._proc_device: Optional[torch.device] = None
        self._last_hs: Optional[torch.Tensor] = None
        self._decoder_handle = None
        self._last_memory: Optional[torch.Tensor] = None
        self._last_spatial_shapes: Optional[torch.Tensor] = None
        self._last_level_start_index: Optional[torch.Tensor] = None
        self._encoder_handle = None

    def _sam3_submodule(self, part: str) -> nn.Module:
        bb = self.sam_model.backbone
        if part == "neck":
            return bb.vision_backbone.convs
        if part == "encoder":
            return self.sam_model.transformer.encoder
        if part == "decoder":
            return self.sam_model.transformer.decoder
        raise ValueError(f"unknown sam3_trainable part {part!r}; "
                         "expected one of neck/encoder/decoder")

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
        capture_memory = self.pixel_source == "sam3_memory" and not self._grad_grounding
        if capture_memory and self._encoder_handle is None:
            self._encoder_handle = install_encoder_hook(
                self.sam_model.transformer.encoder, self._capture_memory,
            )

    def _capture_memory(self, out: dict):
        self._last_memory = out["memory"].detach()
        self._last_spatial_shapes = out["spatial_shapes"]
        self._last_level_start_index = out["level_start_index"]

    def _memory_map(self, mem, shapes, starts) -> torch.Tensor:
        # (sum_hw, B, 256) -> (B, 256, h, w) for the finest level; reshape only.
        # mem keeps grad on the grad-grounding path, so do not detach here.
        mem = mem.float()
        hw = [(int(h), int(w)) for h, w in shapes]
        fi = max(range(len(hw)), key=lambda i: hw[i][0] * hw[i][1])
        h, w = hw[fi]
        s = int(starts[fi])
        return mem[s:s + h * w].permute(1, 2, 0).reshape(mem.shape[1], 256, h, w)

    def _forward_image_grad(self, img: torch.Tensor) -> dict:
        bb = self.sam_model.backbone
        neck = bb.vision_backbone
        if self._lora:
            xs = neck.trunk(img)
            x = getattr(xs[-1], "tensors", xs[-1])
        else:
            with torch.no_grad():
                xs = neck.trunk(img)
            x = getattr(xs[-1], "tensors", xs[-1]).detach()
        sam3_features, sam3_pos = [], []
        for conv in neck.convs:
            o = conv(x)
            sam3_features.append(o)
            sam3_pos.append(neck.position_encoding(o).to(o.dtype))
        scalp = int(getattr(bb, "scalp", 0) or 0)
        if scalp > 0:
            sam3_features, sam3_pos = sam3_features[:-scalp], sam3_pos[:-scalp]
        return {
            "vision_features": sam3_features[-1],
            "vision_pos_enc": sam3_pos,
            "backbone_fpn": sam3_features,
            "sam2_backbone_out": None,
        }

    def _run_once_grad(self, letterboxed: torch.Tensor):
        assert self.prompt_mode != "singleclass", (
            "grad grounding (sam3_trainable) currently supports multiclass prompts only"
        )
        device = letterboxed.device
        img = v2.functional.to_image(letterboxed).to(device)
        img = self._proc.transform(img).unsqueeze(0)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            backbone_out = self._forward_image_grad(img)
            with torch.no_grad():
                text_out = self.sam_model.backbone.forward_text(
                    [" . ".join(self.prompt_classes)], device=device,
                )
            backbone_out.update(text_out)
            out = self.sam_model.forward_grounding(
                backbone_out=backbone_out,
                find_input=self._proc.find_stage,
                geometric_prompt=self.sam_model._get_dummy_prompt(),
                find_target=None,
            )
        queries = out["queries"][0].float()
        if self.pixel_source == "sam3_memory":
            eo = out["prev_encoder_out"]["encoder_out"]
            mem_map = self._memory_map(
                out["encoder_hidden_states"], eo["spatial_shapes"], eo["level_start_index"],
            )
            return [mem_map], queries
        return [t.float() for t in backbone_out["backbone_fpn"]], queries

    def _run_once(self, raw_image: torch.Tensor):
        device_type = raw_image.device.type
        self._last_hs = None
        self._last_memory = None
        raw_image = letterbox_to_square(raw_image, size=self.letterbox_size)
        if self._grad_grounding:
            return self._run_once_grad(raw_image)
        return self._run_once_frozen(raw_image, device_type)

    def _run_once_frozen(self, raw_image: torch.Tensor, device_type: str):
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

        n_levels = len(self.embed_dims)
        if self.pixel_source == "sam3_memory":
            assert self._last_memory is not None, "fusion-encoder hook did not fire"
            mem_map = self._memory_map(
                self._last_memory, self._last_spatial_shapes, self._last_level_start_index,
            )
            return [mem_map], queries
        fpn_raw = state["backbone_out"]["backbone_fpn"]
        assert len(fpn_raw) >= n_levels
        return [t.float() for t in list(fpn_raw)[:n_levels]], queries

    def forward(self, image: torch.Tensor):
        if self._freeze_sam3:
            self.sam_model.eval()
            if self._lora:
                self.sam_model.backbone.vision_backbone.trunk.train()
        device = image.device
        self._ensure_processor(device)
        denorm = denormalize_imagenet(image)

        B = image.shape[0]

        per_level: Optional[List[List[torch.Tensor]]] = None
        per_query: List[torch.Tensor] = []
        for b in range(B):
            feats_b, queries = self._run_once(denorm[b])
            if per_level is None:
                per_level = [[] for _ in feats_b]
            for i, f in enumerate(feats_b):
                per_level[i].append(f if f.dim() == 4 else f.unsqueeze(0))
            per_query.append(queries)
        stacked = [torch.cat(p, dim=0) for p in per_level]

        if self.pixel_source == "sam3_memory":
            feats = list(self.memory_fpn(stacked[0]))
        else:
            feats = list(stacked)

        max_k = max(q.shape[0] for q in per_query)
        padded = torch.zeros(B, max_k, 256, device=device, dtype=torch.float32)
        for b, q in enumerate(per_query):
            padded[b, : q.shape[0]] = q

        return (*feats, padded)

    def output_crop_geometry(self, original_shape):
        # Content-band fractions of the letterboxed square; IDisc.forward crops
        # the prediction to this band before resizing to the original geometry.
        return letterbox_geometry(original_shape, size=self.letterbox_size)
