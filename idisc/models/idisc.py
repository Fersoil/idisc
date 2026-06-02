"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)

Top-level `IDisc` model. Composes pixel_encoder → pixel_decoder → AFP/IDR → ISD.
`sam_mode` selects the IDR source when SAM3 queries are present:
  - None          — encoder produces no queries; IDRs come from AFP (baseline).
  - "linear_proj" — IDRs come from a Linear projection of SAM3 queries (sam3_proj).
  - "adapter"     — IDRs are SAM3 queries enriched with FPN context (ContextAdapter).
`IDisc.build(config)` is the canonical constructor — do not call __init__ directly.
"""

import warnings
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from idisc.models.defattn_decoder import MSDeformAttnPixelDecoder
from idisc.models.fpn_decoder import BasePixelDecoder
from idisc.models.id_module import AFP, ContextAdapter, ISD

SAM3_D_MODEL = 256


class IDisc(nn.Module):
    def __init__(
        self,
        pixel_encoder: nn.Module,
        afp: nn.Module,
        pixel_decoder: nn.Module,
        isd: nn.Module,
        loss: nn.Module,
        afp_min_resolution=1,
        eps: float = 1e-6,
        num_resolutions: int = 3,
        latent_dim: int = 128,
        context_adapter: Optional[nn.Module] = None,
        **kwargs
    ):
        super().__init__()
        self.eps = eps
        self.pixel_encoder = pixel_encoder
        self.afp = afp
        self.pixel_decoder = pixel_decoder
        self.isd = isd
        self.afp_min_resolution = afp_min_resolution
        self.loss = loss
        self.num_resolutions = num_resolutions

        self.sam3_proj = nn.ModuleList([
            nn.Linear(SAM3_D_MODEL, latent_dim)
            for _ in range(num_resolutions)
        ])
        self.context_adapter = context_adapter

    def invert_encoder_output_order(
        self, xs: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        return tuple(xs[::-1])

    def filter_decoder_relevant_resolutions(
        self, decoder_outputs: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        return tuple(decoder_outputs[self.afp_min_resolution :])

    def forward(
        self,
        image: torch.Tensor,
        instance_queries=None,
        sam_mode=None,
        gt: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        pre_extracted_encoder_outputs: Optional[Tuple[torch.Tensor, ...]] = None,
    ):
        """
        Args:
            instance_queries: SAM3 query embeddings (K, 256) for sam_mode in
                {linear_proj, adapter}. If None, IDRs come from AFP.
            sam_mode: None | "linear_proj" | "adapter".
            pre_extracted_encoder_outputs: If provided, skip the pixel_encoder
                call. Used by the video encoder training loop to avoid
                double-running the backbone.
        """
        losses = {"opt": {}, "stat": {}}
        original_shape = gt.shape[-2:] if gt is not None else image.shape[-2:]

        if pre_extracted_encoder_outputs is not None:
            encoder_outputs = pre_extracted_encoder_outputs
        else:
            encoder_outputs = self.pixel_encoder(image)
            if getattr(self.pixel_encoder, "yields_instance_queries", False):
                *encoder_outputs, encoder_queries = encoder_outputs
                encoder_outputs = tuple(encoder_outputs)
                if instance_queries is None:
                    instance_queries = encoder_queries
            encoder_outputs = self.invert_encoder_output_order(encoder_outputs)

        if self.pixel_decoder is not None:
            fpn_outputs, decoder_outputs = self.pixel_decoder(encoder_outputs)
        else:
            # pixel_source in {sam3_memory, backbone_fpn}: encoder features feed
            # ISD directly (no iDisc pixel decoder).
            fpn_outputs = decoder_outputs = tuple(encoder_outputs)
        decoder_outputs = self.filter_decoder_relevant_resolutions(decoder_outputs)
        fpn_outputs = self.filter_decoder_relevant_resolutions(fpn_outputs)

        if instance_queries is not None and instance_queries.shape[0] > 0:
            iq = instance_queries.unsqueeze(0) if instance_queries.dim() == 2 else instance_queries
            if sam_mode == "adapter":
                if self.context_adapter is None:
                    raise RuntimeError(
                        "sam_mode='adapter' requires context_adapter to be built"
                    )
                idrs = self.context_adapter(iq, decoder_outputs)
            else:  # "linear_proj": one Linear per resolution over the queries
                idrs = tuple(proj(iq) for proj in self.sam3_proj)
        else:
            idrs = self.afp(decoder_outputs)
        outs = self.isd(fpn_outputs, idrs)

        out_lst = []
        for out in outs:
            if out.shape[1] == 1:
                out = F.interpolate(
                    torch.exp(out),
                    size=outs[-1].shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
            else:
                out = self.normalize_normals(
                    F.interpolate(
                        out,
                        size=outs[-1].shape[-2:],
                        mode="bilinear",
                        align_corners=True,
                    )
                )
            out_lst.append(out)

        out = torch.mean(torch.stack(out_lst, dim=0), dim=0)
        # Crop the content band out of a letterboxed prediction before resizing.
        # Encoders that don't letterbox lack output_crop_geometry -> full-frame.
        geom = getattr(self.pixel_encoder, "output_crop_geometry", None)
        ft, fl, fh, fw = (
            geom(original_shape) if geom is not None else (0.0, 0.0, 1.0, 1.0)
        )
        _, _, H, W = out.shape
        top, left = round(ft * H), round(fl * W)
        h_c, w_c = max(1, round(fh * H)), max(1, round(fw * W))
        out = out[:, :, top:top + h_c, left:left + w_c]
        out = F.interpolate(
            out,
            original_shape,
            # Legacy code for reproducibility for normals...
            mode="bilinear" if out.shape[1] == 1 else "bicubic",
            align_corners=True,
        )
        if gt is not None:
            losses["opt"] = {
                self.loss.name: self.loss.weight
                * self.loss(out, target=gt, mask=mask.bool(), interpolate=True)
            }
        return (
            out if out.shape[1] == 1 else out[:, :3],
            losses,
            {"outs": outs, "queries": idrs},
        )

    def normalize_normals(self, norms):
        min_kappa = 0.01
        norm_x, norm_y, norm_z, kappa = torch.split(norms, 1, dim=1)
        norm = torch.sqrt(norm_x**2.0 + norm_y**2.0 + norm_z**2.0 + 1e-6)
        kappa = F.elu(kappa) + 1.0 + min_kappa
        norms = torch.cat([norm_x / norm, norm_y / norm, norm_z / norm, kappa], dim=1)
        return norms

    def load_pretrained(self, model_file):
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        dict_model = torch.load(model_file, map_location=device)
        new_state_dict = deepcopy(
            {k.replace("module.", ""): v for k, v in dict_model.items()}
        )
        # strict=False is intentional: the frozen SAM3 backbone is rebuilt from
        # sam_checkpoint at construction, so its keys are never in a head
        # checkpoint and show up as (expected) missing keys. We warn about the
        # two cases that actually signal a mismatch.
        info = self.load_state_dict(new_state_dict, strict=False)
        if info.unexpected_keys:
            # Checkpoint weights with no home in this model — a red flag
            # (architecture/checkpoint drift), so list them in full.
            warnings.warn(
                f"load_pretrained({model_file}): {len(info.unexpected_keys)} "
                f"unexpected key(s) ignored: {info.unexpected_keys}",
                stacklevel=2,
            )

        frozen_prefixes = ("pixel_encoder.sam_model.", "pixel_encoder.video_model.")
        head_missing = [k for k in info.missing_keys
                        if not k.startswith(frozen_prefixes)]
        if head_missing:
            # A non-backbone missing key is a trainable param left at random init.
            # Group by top-level module so e.g. unused sam3_proj in the baseline
            # is distinguishable from a genuinely dropped pixel_decoder/isd weight.
            by_module: Dict[str, int] = {}
            for k in head_missing:
                top = k.split(".", 1)[0]
                by_module[top] = by_module.get(top, 0) + 1
            summary = ", ".join(f"{m}: {n}" for m, n in sorted(by_module.items()))
            warnings.warn(
                f"load_pretrained({model_file}): {len(head_missing)} non-backbone "
                f"key(s) left at init (per module: {summary}). These are trainable "
                f"params the checkpoint did not provide — confirm this is intended.",
                stacklevel=2,
            )

    @property
    def device(self):
        return next(self.parameters()).device

    @classmethod
    def build(cls, config: Dict[str, Dict[str, Any]]):
        # Copy-on-write: build() writes derived values (e.g. embed_dims) back into
        # the config so the sub-builders below can read them, but the caller's
        # config is the resolved source-of-truth that gets snapshotted — do not
        # mutate it.
        config = deepcopy(config)
        pixel_encoder_img_size = config["model"]["pixel_encoder"]["img_size"]
        pixel_encoder_pretrained = config["model"]["pixel_encoder"].get(
            "pretrained", None
        )
        config_backone = {"img_size": np.array(pixel_encoder_img_size)}
        if pixel_encoder_pretrained is not None:
            config_backone["pretrained"] = pixel_encoder_pretrained
        for extra_key in (
            "sam_checkpoint",
            "prompt_mode",
            "prompt_classes",
            "freeze_sam3",
            "load_from_HF",
            "use_presence_score",
            "confidence_threshold",
            "pixel_source",
            "sam3_trainable",
            "lora",
        ):
            if extra_key in config["model"]["pixel_encoder"]:
                config_backone[extra_key] = config["model"]["pixel_encoder"][extra_key]
        import importlib

        mod = importlib.import_module("idisc.models.encoder")
        pixel_encoder_factory = getattr(mod, config["model"]["pixel_encoder"]["name"])
        pixel_encoder = pixel_encoder_factory(**config_backone)

        pixel_encoder_embed_dims = getattr(pixel_encoder, "embed_dims")
        config["model"]["pixel_encoder"]["embed_dims"] = pixel_encoder_embed_dims

        # sam3_memory and backbone_fpn feed ISD directly, so no iDisc pixel decoder.
        pixel_source = config["model"]["pixel_encoder"].get("pixel_source", "msda")
        if pixel_source in ("sam3_memory", "backbone_fpn"):
            pixel_decoder = None
        else:
            pixel_decoder = (
                MSDeformAttnPixelDecoder.build(config)
                if config["model"]["attn_dec"]
                else BasePixelDecoder.build(config)
            )
        afp = AFP.build(config)
        isd = ISD.build(config)

        context_adapter = (
            ContextAdapter.build(config)
            if config.get("method", {}).get("sam_mode") == "adapter"
            else None
        )

        mod = importlib.import_module("idisc.optimization.losses")
        loss = getattr(mod, config["training"]["loss"]["name"]).build(config)

        return deepcopy(
            cls(
                pixel_encoder=pixel_encoder,
                pixel_decoder=pixel_decoder,
                afp=afp,
                isd=isd,
                loss=loss,
                afp_min_resolution=len(pixel_encoder_embed_dims)
                - config["model"]["isd"]["num_resolutions"],
                num_resolutions=config["model"]["isd"]["num_resolutions"],
                latent_dim=config["model"]["afp"]["latent_dim"],
                context_adapter=context_adapter,
            )
        )
