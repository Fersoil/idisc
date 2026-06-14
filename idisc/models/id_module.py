"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from idisc.utils import (AttentionLayer, PositionEmbeddingSine,
                         _get_activation_cls, get_norm)


def mask_pool_centers(masks, feature_map, eps=1e-6):
    """Mask-weighted average pool of `feature_map` under each mask → (B, K, C).

    Used to seed the mask_adapter tokens before the learnable masked
    cross-attention refines them. `masks` are SAM3 mask logits (B, K, H, W).
    """
    m = torch.sigmoid(
        F.interpolate(masks, size=feature_map.shape[-2:],
                      mode="bilinear", align_corners=False)
    )
    if not masks.requires_grad:
        m = m.detach()
    mf = m.flatten(2)                              # (B, K, hw)
    xf = feature_map.flatten(2).transpose(1, 2)    # (B, hw, C)
    return (mf @ xf) / (mf.sum(-1, keepdim=True) + eps)


class ISDHead(nn.Module):
    def __init__(
        self,
        depth: int,
        pixel_dim: int = 256,
        query_dim: int = 256,
        num_heads: int = 4,
        output_dim: int = 1,
        expansion: int = 2,
        activation: str = "silu",
        norm: str = "LN",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.depth = depth
        self.eps = eps
        self.pixel_pe = PositionEmbeddingSine(pixel_dim // 2, normalize=True)
        for i in range(self.depth):
            setattr(
                self,
                f"cross_attn_{i+1}",
                AttentionLayer(
                    sink_dim=pixel_dim,
                    hidden_dim=pixel_dim,
                    source_dim=query_dim,
                    output_dim=pixel_dim,
                    num_heads=num_heads,
                    dropout=0.0,
                    pre_norm=True,
                    sink_competition=False,
                ),
            )
            setattr(
                self,
                f"mlp_{i+1}",
                nn.Sequential(
                    get_norm(norm, pixel_dim),
                    nn.Linear(pixel_dim, expansion * pixel_dim),
                    _get_activation_cls(activation),
                    nn.Linear(expansion * pixel_dim, pixel_dim),
                ),
            )
        self.value_proj = nn.Linear(query_dim, pixel_dim)
        setattr(
            self,
            "proj_output",
            nn.Sequential(
                get_norm(norm, pixel_dim),
                nn.Linear(pixel_dim, pixel_dim),
                get_norm(norm, pixel_dim),
                nn.Linear(pixel_dim, output_dim),
            ),
        )

    def forward(self, feature_map, idrs=None, assign=None, centers=None):
        b, c, h, w = feature_map.shape
        feature_map = rearrange(
            feature_map + self.pixel_pe(feature_map), "b c h w -> b (h w) c"
        )

        if assign is not None:
            feature_map = feature_map + assign @ self.value_proj(centers)
            for i in range(self.depth):
                feature_map = feature_map + getattr(self, f"mlp_{i+1}")(feature_map.clone())
        else:
            for i in range(self.depth):
                update = getattr(self, f"cross_attn_{i+1}")(feature_map.clone(), idrs)
                feature_map = feature_map + update
                feature_map = feature_map + getattr(self, f"mlp_{i+1}")(feature_map.clone())
        out = getattr(self, "proj_output")(feature_map)
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)

        return out


class ISD(nn.Module):
    def __init__(
        self,
        num_resolutions,
        depth,
        pixel_dim=128,
        query_dim=128,
        num_heads: int = 4,
        output_dim: int = 1,
        expansion: int = 2,
        activation: str = "silu",
        norm: str = "torchLN",
    ):
        super().__init__()
        self.num_resolutions = num_resolutions
        for i in range(num_resolutions):
            setattr(
                self,
                f"head_{i+1}",
                ISDHead(
                    depth=depth,
                    pixel_dim=pixel_dim,
                    query_dim=query_dim,
                    num_heads=num_heads,
                    output_dim=output_dim,
                    expansion=expansion,
                    activation=activation,
                    norm=norm,
                ),
            )

    def forward(self, xs, idrs=None, masks=None):
        outs = []
        for i in range(self.num_resolutions):
            if masks is not None:
                m = torch.sigmoid(
                    F.interpolate(masks, size=xs[i].shape[-2:],
                                  mode="bilinear", align_corners=False)
                )
                if not masks.requires_grad:
                    m = m.detach()
                mf = m.flatten(2)
                xf = xs[i].flatten(2).transpose(1, 2)
                centers = (mf @ xf) / (mf.sum(-1, keepdim=True) + 1e-6)
                assign = (mf / (mf.sum(1, keepdim=True) + 1e-6)).transpose(1, 2)
                out = getattr(self, f"head_{i+1}")(xs[i], assign=assign, centers=centers)
            else:
                out = getattr(self, f"head_{i+1}")(xs[i], idrs[i])
            outs.append(out)
        return tuple(outs)

    @classmethod
    def build(cls, config):
        obj = cls(
            num_resolutions=config["model"]["isd"]["num_resolutions"],
            depth=config["model"]["isd"]["depths"],
            pixel_dim=config["model"]["pixel_decoder"]["hidden_dim"],
            query_dim=config["model"]["afp"]["latent_dim"],
            output_dim=config["model"]["output_dim"],
            num_heads=config["model"]["num_heads"],
            expansion=config["model"]["expansion"],
            activation=config["model"]["activation"],
        )
        return obj


class AFP(nn.Module):
    def __init__(
        self,
        num_resolutions: int,
        depth: int = 3,
        pixel_dim: int = 256,
        latent_dim: int = 256,
        num_latents: int = 128,
        num_heads: int = 4,
        activation: str = "silu",
        norm: str = "torchLN",
        expansion: int = 2,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_resolutions = num_resolutions
        self.iters = depth
        self.num_slots = num_latents
        self.latent_dim = latent_dim
        self.pixel_dim = pixel_dim
        self.eps = eps

        bottleneck_dim = expansion * latent_dim
        for i in range(self.num_resolutions):
            setattr(
                self,
                f"pixel_pe_{i+1}",
                PositionEmbeddingSine(pixel_dim // 2, normalize=True),
            )
            setattr(
                self,
                f"mu_{i+1}",
                nn.Parameter(torch.randn(1, self.num_slots, latent_dim)),
            )

        # Set up attention iterations
        for j in range(self.iters):
            for i in range(self.num_resolutions):
                setattr(
                    self,
                    f"cross_attn_{i+1}_d{1}",
                    AttentionLayer(
                        sink_dim=latent_dim,
                        hidden_dim=latent_dim,
                        source_dim=pixel_dim,
                        output_dim=latent_dim,
                        num_heads=num_heads,
                        dropout=0.0,
                        pre_norm=True,
                        sink_competition=True,
                    ),
                )
                setattr(
                    self,
                    f"mlp_cross_{i+1}_d{1}",
                    nn.Sequential(
                        get_norm(norm, latent_dim),
                        nn.Linear(latent_dim, bottleneck_dim),
                        _get_activation_cls(activation),
                        nn.Linear(bottleneck_dim, latent_dim),
                    ),
                )

    def forward(
        self, feature_maps: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        b, *_ = feature_maps[0].shape
        idrs = []
        feature_maps_flat = []
        for i in range(self.num_resolutions):
            # feature maps embedding pre-process
            feature_map, (h, w) = feature_maps[i], feature_maps[i].shape[-2:]
            feature_maps_flat.append(
                rearrange(
                    feature_map + getattr(self, f"pixel_pe_{i+1}")(feature_map),
                    "b d h w-> b (h w) d",
                )
            )
            # IDRs generation
            idrs.append(getattr(self, f"mu_{i+1}").expand(b, -1, -1))

        # layers
        for i in range(self.num_resolutions):
            for _ in range(self.iters):
                # Cross attention ops
                idrs[i] = idrs[i] + getattr(self, f"cross_attn_{i+1}_d{1}")(
                    idrs[i].clone(), feature_maps_flat[i]
                )
                idrs[i] = idrs[i] + getattr(self, f"mlp_cross_{i+1}_d{1}")(
                    idrs[i].clone()
                )

        return tuple(idrs)

    @classmethod
    def build(cls, config):
        output_num_resolutions = (
            len(config["model"]["pixel_encoder"]["embed_dims"])
            - config["model"]["afp"]["context_low_resolutions_skip"]
        )
        obj = cls(
            num_resolutions=output_num_resolutions,
            depth=config["model"]["afp"]["depths"],
            pixel_dim=config["model"]["pixel_decoder"]["hidden_dim"],
            num_latents=config["model"]["afp"]["num_latents"],
            latent_dim=config["model"]["afp"]["latent_dim"],
            num_heads=config["model"]["num_heads"],
            expansion=config["model"]["expansion"],
            activation=config["model"]["activation"],
        )
        return obj


class ContextAdapter(nn.Module):
    """IDR source for sam_mode="adapter": per ISD resolution the SAM3 tokens
    self-attend, cross-attend over that level's feature map, then run an MLP,
    for `depth` iterations. Output is `(B, K, dim)` per resolution."""

    def __init__(
        self,
        num_resolutions: int,
        depth: int = 2,
        dim: int = 256,
        pixel_dim: int = 256,
        num_heads: int = 4,
        activation: str = "silu",
        norm: str = "torchLN",
        expansion: int = 2,
        mask_attend: bool = True,
    ):
        super().__init__()
        self.num_resolutions = num_resolutions
        self.iters = depth
        self.dim = dim
        # When fed a per-token support mask, restrict each token's cross-attention
        # to its own object's pixels (learnable masked pooling). Off => the tokens
        # cross-attend over the whole feature map (unmasked control).
        self.mask_attend = mask_attend
        bottleneck_dim = expansion * dim
        for i in range(num_resolutions):
            setattr(
                self,
                f"pixel_pe_{i+1}",
                PositionEmbeddingSine(pixel_dim // 2, normalize=True),
            )
            for j in range(depth):
                setattr(
                    self,
                    f"self_attn_{i+1}_{j+1}",
                    AttentionLayer(
                        sink_dim=dim, hidden_dim=dim, source_dim=None,
                        output_dim=dim, num_heads=num_heads, dropout=0.0,
                        pre_norm=True, sink_competition=False,
                    ),
                )
                setattr(
                    self,
                    f"cross_attn_{i+1}_{j+1}",
                    AttentionLayer(
                        sink_dim=dim, hidden_dim=dim, source_dim=pixel_dim,
                        output_dim=dim, num_heads=num_heads, dropout=0.0,
                        pre_norm=True, sink_competition=False,
                    ),
                )
                setattr(
                    self,
                    f"mlp_{i+1}_{j+1}",
                    nn.Sequential(
                        get_norm(norm, dim),
                        nn.Linear(dim, bottleneck_dim),
                        _get_activation_cls(activation),
                        nn.Linear(bottleneck_dim, dim),
                    ),
                )

    def forward(
        self,
        tokens: torch.Tensor,
        feature_maps: Tuple[torch.Tensor, ...],
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        idrs = []
        for i in range(self.num_resolutions):
            fmap = feature_maps[i]
            ctx = rearrange(
                fmap + getattr(self, f"pixel_pe_{i+1}")(fmap), "b d h w -> b (h w) d"
            )
            attn_bias = None
            if self.mask_attend and masks is not None:
                # Hard support mask: token k may only pool from pixels where mask k
                # is active; the attention weights inside are learned (not a log
                # bias, not an average). Empty masks fall back to full attention.
                m = torch.sigmoid(
                    F.interpolate(masks, size=fmap.shape[-2:],
                                  mode="bilinear", align_corners=False)
                )
                if not masks.requires_grad:
                    m = m.detach()
                keep = m.flatten(2) >= 0.5               # (b, K, hw)
                keep[keep.sum(-1) == 0] = True
                attn_bias = torch.where(
                    keep, 0.0, torch.tensor(-1e4)
                ).to(ctx.dtype)
            x = tokens
            for j in range(self.iters):
                x = x + getattr(self, f"self_attn_{i+1}_{j+1}")(x.clone())
                x = x + getattr(self, f"cross_attn_{i+1}_{j+1}")(
                    x.clone(), ctx, attn_bias=attn_bias
                )
                x = x + getattr(self, f"mlp_{i+1}_{j+1}")(x.clone())
            idrs.append(x)
        return tuple(idrs)

    @classmethod
    def build(cls, config):
        adapter_cfg = config["model"].get("context_adapter", {}) or {}
        return cls(
            num_resolutions=config["model"]["isd"]["num_resolutions"],
            depth=adapter_cfg.get("depth", 2),
            dim=config["model"]["afp"]["latent_dim"],
            pixel_dim=config["model"]["pixel_decoder"]["hidden_dim"],
            num_heads=config["model"]["num_heads"],
            expansion=config["model"]["expansion"],
            activation=config["model"]["activation"],
            mask_attend=adapter_cfg.get("mask_attend", True),
        )


class MemoryFPN(nn.Module):
    """Learned SimpleFPN turning SAM3's single fusion `memory` level into an
    n-level pyramid with ConvTranspose (up) / conv (same) / strided conv (down).
    Level i has scale `top_scale / 2**i`: 72x72 input, top_scale=2 -> [144^2,72^2,36^2];
    top_scale=4 -> [288^2,144^2,72^2] (matches backbone_fpn resolution)."""

    def __init__(self, dim: int = 256, n_levels: int = 3, num_groups: int = 8,
                 top_scale: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(n_levels):
            scale = top_scale / (2.0 ** i)
            layers = []
            if scale > 1.0:
                f = int(scale)
                layers += [nn.ConvTranspose2d(dim, dim, kernel_size=f, stride=f), nn.GELU()]
            elif scale < 1.0:
                f = int(round(1.0 / scale))
                layers += [nn.Conv2d(dim, dim, kernel_size=f, stride=f), nn.GELU()]
            layers += [
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups, dim),
                nn.GELU(),
            ]
            self.blocks.append(nn.Sequential(*layers))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return tuple(blk(x) for blk in self.blocks)
