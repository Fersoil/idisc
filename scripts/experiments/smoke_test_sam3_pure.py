#!/usr/bin/env python
"""Smoke test for the pure SAM3 + iDisc d2c configuration.

Validates that:
  1. Hydra runtime config resolves to variant 'sam-replace'.
  2. IDisc.build assembles fresh weights with pixel_encoder='sam3_image'.
  3. The encoder runs SAM3 once and returns (4 features + instance_queries).
  4. A full model forward produces a depth map at the GT resolution.
  5. One backward + optimizer step updates ISD weights but leaves SAM3 frozen.
  6. The SAM3 backbone hook fires exactly once per encoder forward (single pass).

Run on GPU:
    python scripts/experiments/smoke_test_sam3_pure.py \
        --config-file configs/kitti/kitti_sam3.json \
        --img-h 352 --img-w 1216
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import optim


def _build_model(config_path: Path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from idisc.models.idisc import IDisc

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    assert config["model"]["pixel_encoder"]["name"] == "sam3_image", (
        "Expected pixel_encoder.name == 'sam3_image'"
    )

    model = IDisc.build(config)
    return model, config


def _check_runtime_config(config_path: Path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from idisc.utils.config_bridge import build_runtime_config

    hydra_cfg = {
        "dataset": {"legacy_config_path": str(config_path)},
        "method": {
            "idr_source": "afp",
            "sam_mode": "replace",
            "query_source": "online",
            "prompt": {"mode": "multiclass", "classes": [], "use_bbox": False, "strategy": "x"},
        },
    }
    runtime = build_runtime_config(hydra_cfg, repo_root=Path(__file__).resolve().parents[2])
    assert runtime["variant"] == "sam-replace", f"Expected sam-replace, got {runtime['variant']}"
    print(f"  [ok] runtime variant = {runtime['variant']}")


def _check_encoder_shapes(model, image: torch.Tensor):
    encoder = model.pixel_encoder
    assert getattr(encoder, "yields_instance_queries", False), \
        "encoder must advertise yields_instance_queries=True"
    n_levels = len(encoder.embed_dims)
    assert encoder.embed_dims == [256] * n_levels, \
        f"unexpected embed_dims: {encoder.embed_dims}"
    print(f"  [ok] encoder advertises {n_levels} FPN levels")

    out = encoder(image)
    assert len(out) == n_levels + 1, \
        f"expected {n_levels}+1-tuple (features + queries), got {len(out)}"
    *feats, queries = out
    for i, f in enumerate(feats):
        assert f.dim() == 4, f"level {i} not 4D: {f.shape}"
        assert f.shape[1] == 256, f"level {i} channel != 256: {f.shape}"
        assert f.shape[0] == image.shape[0], f"level {i} batch mismatch: {f.shape}"
    h_w = [(f.shape[-2], f.shape[-1]) for f in feats]
    assert all(h_w[i][0] >= h_w[i + 1][0] and h_w[i][1] >= h_w[i + 1][1] for i in range(n_levels - 1)), \
        f"feature levels not monotonically decreasing: {h_w}"
    assert queries.dim() == 3 and queries.shape[-1] == 256, \
        f"queries shape wrong: {queries.shape}"
    print(f"  [ok] feature pyramid HxW: {h_w}")
    print(f"  [ok] instance_queries: {tuple(queries.shape)}")


def _check_freeze(model):
    sam_params = list(model.pixel_encoder.sam_model.parameters())
    n_frozen = sum(1 for p in sam_params if not p.requires_grad)
    assert n_frozen == len(sam_params), \
        f"SAM3 should be fully frozen; found {len(sam_params) - n_frozen} trainable params"
    trainable_external = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"  [ok] SAM3 frozen ({len(sam_params)} params); "
          f"{trainable_external:,} trainable on iDisc side")


def _check_forward_backward(model, image, gt, mask, device):
    pred, losses, _ = model(image, gt=gt, mask=mask, sam_mode="replace")
    assert pred.shape[-2:] == gt.shape[-2:], \
        f"pred/gt spatial mismatch: {pred.shape} vs {gt.shape}"
    assert losses["opt"], "no losses returned"
    print(f"  [ok] forward output shape: {tuple(pred.shape)}")

    # backward
    loss_val = sum(v for v in losses["opt"].values())
    loss_val.backward()
    assert torch.isfinite(loss_val).item(), f"non-finite loss: {loss_val}"
    print(f"  [ok] loss = {loss_val.item():.4f}")

    proj_grad = model.sam3_proj[0].weight.grad
    assert proj_grad is not None, "sam3_proj.weight.grad is None"
    assert torch.isfinite(proj_grad).all(), "sam3_proj.weight.grad has non-finite values"
    print(f"  [ok] sam3_proj.weight.grad |.mean| = {proj_grad.abs().mean():.4e}")

    sam_grads = [p.grad for p in model.pixel_encoder.sam_model.parameters() if p.grad is not None]
    assert not sam_grads, \
        f"SAM3 has gradients despite being frozen ({len(sam_grads)} params)"
    print(f"  [ok] SAM3 has no grads (frozen)")

    isd_param = next(model.isd.parameters())
    before = isd_param.detach().clone()
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    optimizer.step()
    moved = (isd_param.detach() - before).abs().max().item()
    assert moved > 0, "ISD weights did not move after optimizer step"
    print(f"  [ok] ISD weight moved by max={moved:.2e} after optimizer step")


def _check_single_pass(model, image):
    model.pixel_encoder._call_counter = 0
    _ = model.pixel_encoder(image)
    n_calls = model.pixel_encoder._call_counter
    assert n_calls == image.shape[0], \
        f"expected {image.shape[0]} backbone calls (one per batch item), got {n_calls}"
    print(f"  [ok] backbone fired {n_calls} times for batch={image.shape[0]} (no double pass)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="configs/kitti/kitti_sam3.json")
    parser.add_argument("--img-h", type=int, default=352)
    parser.add_argument("--img-w", type=int, default=1216)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / args.config_file
    if not torch.cuda.is_available():
        raise RuntimeError(
            "SAM3 hardcodes device='cuda' in its position encoding and cannot "
            "be instantiated without a GPU. Run this smoke test on a compute "
            "node with CUDA (e.g. via srun --gres=gpu:1)."
        )
    device = torch.device("cuda")

    print(f"Repo root: {repo_root}")
    print(f"Device:    {device}")
    print(f"Config:    {config_path}")

    print("\n[1/6] Runtime config resolution")
    _check_runtime_config(config_path)

    print("\n[2/6] Build IDisc with sam3_image encoder")
    model, _config = _build_model(config_path)
    model = model.to(device)
    print(f"  [ok] model on {device}")

    print("\n[3/6] Encoder shapes")
    image = torch.randn(args.batch_size, 3, args.img_h, args.img_w, device=device)
    _check_encoder_shapes(model, image)

    print("\n[4/6] Freeze enforcement")
    _check_freeze(model)

    print("\n[5/6] Forward + backward + optimizer step")
    gt = torch.rand(args.batch_size, 1, args.img_h, args.img_w, device=device) * 80.0 + 0.5
    mask = torch.ones_like(gt, dtype=torch.bool)
    _check_forward_backward(model, image, gt, mask, device)

    print("\n[6/6] Single-pass guarantee (no double backbone call)")
    _check_single_pass(model, image)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
