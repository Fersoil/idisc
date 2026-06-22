# SAM3Depth — reproducibility map

Every number in the report mapped to the run/method that produced it and the code
that re-runs it. All tooling lives in this tree (`main`).

## Entry points

- Train / eval dispatch: `scripts/run_with_hydra.py` (Hydra) → `scripts/train.py` (`run_train`) / `scripts/experiments/eval_depth.py` (`run_eval`).
- Standalone eval CLI: `scripts/experiments/eval_depth.py --config <resolved_config.yaml> --checkpoint <ckpt.pt> --output-dir <dir>` (`--flip` adds horizontal-flip TTA).
- Cluster launcher: `scripts/launch.sh experiment=<name> [overrides]`.
- Every run writes `output/runs/<ts>_<exp_id>_<sha>/{resolved_config.yaml,manifest.json,metrics.json,stdout.log}`; checkpoints go to `output/models/<exp_id>/`.

## Configuration axes (SAM3 head)

| Axis | Key | Values |
|---|---|---|
| IDR source | `method.sam_mode` | `linear_proj`, `adapter`, `mask_linear` |
| Pixel source | `model.pixel_encoder.pixel_source` | `msda`, `sam3_memory`, `backbone_fpn` |
| Unfreeze | `model.pixel_encoder.sam3_trainable` | `[]` frozen, or subset of `[neck,encoder,decoder]` |
| Iterations | `finetune.n_iters` | 5000 / 15000 |
| Dataset | `dataset` | `kitti`, `nyu` |

Validation gate for legal combinations: `idisc/utils/config_bridge.py:_validate()`.

## Setup numbers (config/data derived)

| Number | Source |
|---|---|
| 23,158 train / 697 test (652 valid) KITTI Eigen | `splits/kitti/kitti_eigen_{train,test}.txt` |
| 654 NYU test | `splits/nyu/nyu_test.txt` |
| 61 drives, 4-frame clips | `splits/kitti/sequence_manifest.json`, `conf/dataset/kitti.yaml` `clip_length: 4` |
| 80 m depth cap | `idisc/dataloders/kitti.py` |
| head LR 5e-5 / SAM3 LR 1e-5, bf16, eff. batch 2 | `conf/finetune/image.yaml` |
| 840 M frozen / 12–49 M trained, d=256 D=2 | model configs |

## Table 1 — depth invariant to the partition's source

AbsRel / δ1 / RMSE. SAM3 rows use `sam3_memory` with the post-trunk unfrozen.

| AbsRel / δ1 / RMSE | Method | Config | Eval run |
|---|---|---|---|
| 0.0600 / 0.9638 / 2.362 | ResNet-101 baseline (released 45k ckpt, re-eval) | `eval_idisc_kitti_image` | `…_eval-idr-off` |
| 0.0691 / 0.9546 / 2.470 | linear, 5k | `finetune_sam3_kitti_linear_mem` | `eval-sam3-linear-idr-off` |
| 0.0618 / 0.9643 / 2.347 | linear, 15k | `finetune_sam3_kitti_linear_mem finetune.n_iters=15000` | `eval-linear_mem_15k` |
| 0.0685 / 0.9548 / 2.459 | adapter, 5k | `finetune_sam3_kitti_adapter_mem` | `eval-sam3-adapter-idr-off` |
| 0.0607 / 0.9652 / 2.310 | adapter, 15k | `finetune_sam3_kitti_adapter_mem finetune.n_iters=15000` | `eval-adapter_mem_15k` |
| 0.0689 / 0.9549 / 2.494 | mask linear, 5k | `finetune_sam3_kitti_mask_linear_mem` | `eval-mask-linear-5k` |
| 0.0619 / 0.9638 / 2.326 | mask linear, 15k | `finetune_sam3_kitti_mask_linear_mem finetune.n_iters=15000` | `eval-mask_linear_mem_15k` |
| 0.078 → 0.073 (AbsRel) | frozen vs unfrozen post-trunk (linear, MSDA) | `finetune_sam3_kitti_linear_msda{,_frozen}` | `…_lp_msda_frozen` / `…_lp_msda_e2e` |

To reproduce a 15k row: train with the config above (`--name <tag>`), then
`scripts/experiments/eval_depth.py --config output/runs/<run>/resolved_config.yaml --checkpoint output/models/<exp_id>/best_sam_finetuned.pt --output-dir output/runs/eval-<tag>`.

**Baseline.** Our in-pipeline re-evaluation of the released `kitti_resnet101.pt`
reproduces the official iDisc result (AbsRel 0.0600 vs 0.059, RMSE 2.362, matching
δ thresholds).

## Table 2 — IDR ablation (zero / swap)

Mechanism: `idisc/models/idisc.py` `_ablate_idrs()`, selected by env `IDR_ABLATE=off|zero|swap|shuffle`; run via `eval_depth.py` under each value. Driver: `scripts/utils/eval_sam3_idr_ablation.sh`.

| Head | off | zero | swap | Run dir |
|---|---|---|---|---|
| SAM3 linear, mem | 0.06913 | 0.18523 | 0.11245 | `output/runs/eval-sam3-linear-idr-{off,zero,swap}` |
| SAM3 adapter, mem | 0.06851 | 0.28730 | 0.11132 | `output/runs/eval-sam3-adapter-idr-*` |
| SAM3 linear, MSDA | 0.07332 | 0.19595 | 0.09391 | `output/runs/eval-sam3-msda-idr-*` |
| iDisc R101, MSDA | 0.05996 | 0.06678 | 0.06122 | `output/runs/*_eval-idr-{off,zero,swap}` |

Shuffle is a no-op (permutation-invariant attention): `*_eval-idr-shuffle` = 0.05996 ≈ off.

## Table 3 — region/depth R²

`scripts/experiments/eval_region_r2.py` (τ=0.5): per image, assign each σ>0.5 pixel to its arg-max SAM3 mask, fit pixel depth to per-region mean depth. Wrapper: `scripts/utils/eval_region_r2.sh`.

| mean R² | median R² | active masks | Run dir |
|---|---|---|---|
| 0.746 | 0.756 | ~196 | `output/runs/r2-frozen/region_r2.json` |
| 0.260 | 0.206 | ~147 | `output/runs/r2-finetuned/region_r2.json` |

JSON schema: `{checkpoint, tau, mean_r2, median_r2, n_images_scored=652, mean_active_masks, elapsed_s}`.

## Table 4 — temporal sweep + flicker

Depth model = released single-image iDisc-R101 (per frame); SAM3 **video tracker**
(`idisc/models/sam3_track.py`) supplies cross-frame instance ids for the loss only.
Loss: `idisc/optimization/grounding_losses.py:temporal_smoothness_loss` (Eq. 5), wired in
`scripts/train.py` as `L_depth + λ·L_tmp`. Flicker: `scripts/experiments/eval_flicker.py`.
Config: `finetune_idisc_kitti_video_temporal finetune.temporal.weight=<λ>`.

| λ | AbsRel | flicker | Run dir |
|---|---|---|---|
| 0 | 0.0575 | ≈0.11 | `…_temporal-l0` |
| 0.1 | 0.0605 | ≈0.11 | `…_temporal-l0.1` |
| 0.3 | 0.0612 | ≈0.11 | `…_temporal-l0.3` |

## NYU (Sec 6)

| AbsRel | Method | Config | Eval dir |
|---|---|---|---|
| 0.1146 | R101 baseline | `finetune_idisc_nyu_image` (eval) | `…_nyu_r101_eval` |
| 0.1212 | SAM3 linear | `finetune_sam3_nyu_linear_mem` | `…_sam3_nyu_linear_eval` |
| 0.1208 | SAM3 mask linear | `finetune_sam3_nyu_mask_linear_mem` | `…_sam3_nyu_mask_linear_eval` |

NYU uses the HuggingFace export; eval protocol = `official`.

## Figures

- Partition comparison → `scripts/vis/visualize_sam3.py`
- Depth + error vs LiDAR → `scripts/vis/visualize_sequence.py`
- Fine-tuning breaks the partition → `scripts/vis/visualize_mask_pool.py`
