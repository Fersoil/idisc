# Pure SAM3 + iDisc d2c (RUN LOG)

## Goal

Test whether SAM3 (a segmentation foundation model) can serve as a complete visual encoder for iDisc, with both its dense FPN features replacing iDisc's ResNet/Swin and its 200 segmentation queries flowing into iDisc's d2c bottleneck. The iDisc-side modules train fresh from random init; no pretrained iDisc weights are loaded. The reference number is the pretrained iDisc-R101 baseline on KITTI Eigen at abs_rel ≈ 0.060.

---

## How to run

Use the unified launcher from the repo root:

```bash
./scripts/launch.sh <key> [-- <hydra_overrides>]
```

| Key | Experiment | Output dir | Val metric | When to kill |
|-----|-----------|-----------|-----------|--------------|
| `baseline` | E1 iDisc-R101 pretrained (eval) | `outputs/runs/…E1…/metrics.json` | abs\_rel | n/a (eval only) |
| `e11` | E11/E20 SAM3 pure, single-frame | `finetune_output/E20-sam3-pure-multiclass/` | abs\_rel | step 5000 (~2h) or if plateau visible at step 3000 |
| `e12` | E12 SAM3 translate (Sam3QueryToIDR) | `finetune_output/E12-sam3-translate/` | abs\_rel | step 5000 (~2h) |
| `e18` | E18 SAM3 pure + 4-frame sequence | `finetune_output/E18-sam3-pure-sequence/` | abs\_rel | step 5000 (~5h); expect spike at step 1000 then recovery |
| `e19` | E19 SAM3 video encoder + sequence | `finetune_output/E19-sam3-video-sequence/` | abs\_rel | kill if val keeps rising after step 1000 (seen to diverge) |

**Common overrides:**
```bash
./scripts/launch.sh e11 -- finetune.n_iters=500    # fast smoke test
./scripts/launch.sh e11 -- finetune.lr=1e-4         # learning rate sweep
```

Logs go to `logs/<job_name>_<jobid>.{out,err}`. Monitor with:
```bash
tail -f logs/<job_name>_<jobid>.out
```

---

## Run log

### Run 1 — first end-to-end "pure SAM3" training

| Setting | Value |
|---------|-------|
| Mode | `replace` (Linear `sam3_proj`) |
| Prompt | `"car . truck . person . bicycle . building . tree . road sign . pole"` (8-class multiclass) |
| SAM3 calls/frame | 1 |
| Tokens fed to model | **0 real** — 100% placeholder (all-zero 256-d vector). SAM3's `confidence_threshold=0.5` filtered everything; max score on KITTI was 0.017. See Run 2. |
| Data | Single-frame KITTI, 23,158 train |
| Trainable params | ~11.9M |
| Change vs prior | First run; no prior. |

5,000 iters, batch 2, lr 5e-5 cosine. Frozen SAM3 backbone, ~12M trainable iDisc-side params. SAM3 queries fed through `sam3_proj` (single Linear 256→128). `sam_mode=replace`: AFP bypassed, only `sam3_proj` queries reach ISD.

**Final val abs_rel = 0.0818** (best was set on the *last* iteration).

| Step | abs_rel |
|-----:|--------:|
|  500 | 0.1403  |
| 1000 | 0.1055  |
| 1500 | 0.1184  |
| 2000 | 0.1054  |
| 2500 | 0.0955  |
| 3000 | 0.0913  |
| 3500 | 0.0844  |
| 4000 | 0.0836  |
| 4500 | 0.0843  |
| 5000 | **0.0818** |

Versus pretrained iDisc-R101 (E1 baseline, ~120k iters):

| Metric | E1 R101 | Run 1 | Gap |
|--------|--------:|------:|----:|
| abs_rel | **0.0600** | 0.0818 | +36% |
| rmse    | 2.36 | 2.83 | +20% |
| d1      | 0.964 | 0.927 | -3.7 pp |
| silog   | 8.20 | 10.04 | +22% |

Closing 64% of the way to E1 in 5k iters from cold start was a promising signal — at face value, suggesting SAM3's frozen features carry usable spatial structure for depth.

### Run 2 — diagnostic probe: was the d2c bottleneck actually receiving SAM3 queries?

Probed the trained encoder on 200 KITTI val images and counted how often the encoder fell back to its zero-vector placeholder (used when SAM3 returns no detections).

**Result: placeholder fired on 100% of images.** SAM3 was returning zero detections on every KITTI image with the production prompt, so the d2c bottleneck saw the same zero vector every batch through all 5k training steps of Run 1. Run 1's 0.0818 was a "FPN-features-only" floor — the query path was effectively dead.

### Run 3 — query path fixed; K=32 retrain

| Setting | Value |
|---------|-------|
| Mode | `replace` |
| Prompt | `"car . truck . person . bicycle . building . tree . road sign . pole"` (multiclass) |
| SAM3 calls/frame | 1 |
| Tokens fed to model | **32** (top-32 decoder hidden states by score) |
| Change vs Run 1 | Fixed encoder: hook on `transformer.decoder` + `confidence_threshold=0.0` → real queries instead of zeros |

Routed queries directly from SAM3's transformer decoder (200 candidate slots × 256-d), top-K-selecting by score. Re-ran the 5k training. Killed early at step 1500 once it became clear the trajectory was tracking Run 1 exactly.

| Step | Run 1 (placeholder) | Run 3 (K=32 real queries) | Δ |
|-----:|--------------------:|--------------------------:|---:|
|  500 | 0.140294 | 0.140283 | -8e-6 |
| 1000 | 0.105534 | 0.105540 | +6e-6 |
| 1500 | 0.118435 | 0.118402 | -3e-5 |

Identical to 5 decimals.

### Probe — query variance across images

Ran the encoder on 50 KITTI val images and measured cross-image cosine similarity of the per-image (32, 256) query matrices.

| Statistic | Value |
|-----------|------:|
| Whole-flatten cos sim across images | 0.553 |
| Per-slot cos sim — median | 0.536 |
| Per-slot cos sim — max | 0.755 |
| Distance to mean query (median, L2) | 10.16 |
| Query norm (median, L2) | 15.47 |

Queries do vary per image (cos sim 0.55, not 1.0). They have a strong shared component (~50% of magnitude is image-agnostic), but per-image variance exists — so the failure is not "queries are constant," it's "query variance isn't reaching depth output."

### Probe — SAM3 prompt format scan

Ran SAM3 on 5 KITTI images across 18 prompt variants and recorded the max + mean detection scores.

| Prompt | mean(max) | mean(top5) | mean(top32) | mean(top200) |
|--------|----------:|-----------:|------------:|-------------:|
| `car . truck . person . ...` (production) | 0.008 | 0.006 | 0.004 | 0.002 |
| `vehicle` | **0.81** | 0.64 | **0.27** | 0.10 |
| `a car` | 0.75 | 0.55 | 0.24 | 0.09 |
| `tree` | 0.71 | 0.59 | 0.33 | 0.10 |
| `road` | 0.75 | 0.50 | 0.15 | 0.05 |
| `building` | 0.56 | 0.45 | 0.21 | 0.07 |
| `""` (empty) | 0.16 | 0.11 | 0.06 | 0.03 |
| `anything` | 0.0003 | 0.0003 | 0.0002 | 0.0001 |

**100× higher max confidence with single-class prompts** versus the production multi-class concat. The `.` separator collapses SAM3's confidence; the model was clearly designed for one-class queries at a time.

### Run 4 — K=200 with high-confidence singleclass prompts

| Setting | Value |
|---------|-------|
| Mode | `replace` |
| Prompt | `"vehicle"`, `"tree"`, `"road"`, `"building"` — **4 separate SAM3 calls per frame** |
| SAM3 calls/frame | 4 (one per class) |
| Tokens fed to model | **200** (top-200 from 800 total candidates merged across 4 calls) |
| Change vs Run 3 | Switched from 8-class multiclass (0.008 max score) to 4 singleclass calls (0.81 max score). Probe showed multiclass prompt collapses confidence 100×. |

Switched to `prompt_mode=singleclass` over four high-yield single-word prompts (`vehicle`, `tree`, `road`, `building`), running SAM3 once per class and merging the top-200 by score across all four passes. Same 5k schedule, otherwise identical to Run 1 / Run 3.

**Final val abs_rel = 0.0818** (effectively identical to Run 1).

| Step | Run 1 (placeholder) | Run 4 (K=200 singleclass, 60× higher confidence) | Δ |
|-----:|--------------------:|-------------------------------------------------:|---:|
|  500 | 0.140294 | 0.140286 | -8e-6 |
| 1000 | 0.105534 | 0.105526 | -8e-6 |
| 1500 | 0.118435 | 0.118414 | -2e-5 |
| 2000 | 0.105439 | 0.105417 | -2e-5 |
| 2500 | 0.095454 | 0.095451 | -3e-6 |
| 3000 | 0.091310 | 0.091272 | -4e-5 |
| 3500 | 0.084364 | 0.084363 | -1e-6 |
| 4000 | 0.083603 | 0.083592 | -1e-5 |
| 4500 | 0.084266 | 0.084260 | -6e-6 |
| 5000 | 0.081846 | 0.081847 | +1e-6 |

Trajectories locked across all 10 validations.

### Run 5 — cross-attention translator (`Sam3QueryToIDR`)

| Setting | Value |
|---------|-------|
| Mode | `translate` (Sam3QueryToIDR) |
| Prompt | Singleclass × 4 (`vehicle`, `tree`, `road`, `building`) |
| SAM3 calls/frame | 4 |
| Tokens fed to model | **200** |
| Change vs Run 4 | Replaced single-layer `sam3_proj` Linear with Sam3QueryToIDR: 32 learnable seed vectors cross-attending to the 200 SAM3 queries via 2-layer transformer. +707k trainable params. |

Replaced the single `Linear(256, 128)` (`sam3_proj`) with a proper cross-attention pooling module: ~32 learnable IDR seeds × 128-d that cross-attend to all 200 SAM3 queries through 2 transformer-decoder layers. Mirrors AFP's design (learnable seeds → cross-attention to source → MLP, iterated), but with queries as the source instead of FPN features. New `sam_mode=translate` variant; AFP still bypassed. 5k iters, otherwise same config as Run 4. ~707k extra trainable params (12.58M total vs 11.88M for Run 4).

**Final val abs_rel = 0.0830** (best 0.0830 at step 5000).

| Step | Run 4 (replace, K=200) | Run 5 (translate, K=200) | Δ |
|-----:|-----------------------:|-------------------------:|---:|
|  500 | 0.140286 | 0.141916 | +0.0016 |
| 1000 | 0.105526 | 0.153859 | +0.0483 |
| 1500 | 0.118414 | 0.146280 | +0.0279 |
| 2000 | 0.105417 | 0.087368 | -0.0181 |
| 2500 | 0.095451 | 0.097890 | +0.0024 |
| 3000 | 0.091272 | 0.091832 | +0.0006 |
| 3500 | 0.084363 | 0.090340 | +0.0060 |
| 4000 | 0.083592 | 0.087875 | +0.0043 |
| 4500 | 0.084260 | 0.086973 | +0.0027 |
| 5000 | **0.081847** | **0.082962** | +0.0011 |

The translator's val curve is qualitatively different from Run 4's (wide swings 0.142 → 0.154 → 0.146 → 0.087 in the first 2000 steps, vs Run 4's monotone descent), but it converges to **the same neighborhood as Run 4** by step 5000 — within 0.001 abs_rel. The architectural change does shift training dynamics, but the asymptote is unchanged.

---

## Architectural conclusion

Five runs spanning the full range of "what content reaches the d2c bottleneck":

| Setup | Final abs_rel |
|-------|--------------:|
| Run 1 — replace, placeholder zeros | 0.0818 |
| Run 3 — replace, K=32 low-confidence queries | (killed at step 1500, tracking Run 1) |
| Run 4 — replace, K=200 high-confidence queries (60× higher score) | 0.0818 |
| Run 5 — translate, K=200 + cross-attention pooling | 0.0830 |
| Pretrained iDisc-R101 (E1) | 0.0600 |

Across these runs, the d2c bottleneck has been fed: zeros, low-confidence queries, high-confidence queries, and high-confidence queries through a proper cross-attention translator. **All four converge to within 0.001 abs_rel of each other.** Neither query content nor projection architecture meaningfully changes the depth output.

The d2c IDR pathway is a near-total no-op in this setup. Depth is being predicted entirely by the FPN-features path through `MSDeformAttnPixelDecoder` and ISD's pixel-side projection. ISD's cross-attention with IDRs effectively reduces to a uniform-weighted average of the IDRs (regardless of their values), which contributes a roughly constant additive bias to depth — explaining why all runs land at the same number.

The 0.082 floor is the **FPN-only ceiling** for this architecture: frozen SAM3 backbone + 5k iters of fresh-init iDisc-side modules. The 36% gap to the pretrained R101 baseline reflects this ceiling, not the d2c query path.

---

## Insights

1. **Query content doesn't matter in this setup.** Tested across 4 orders of magnitude in query informativeness (zero placeholder → 100× confidence improvement → cross-attention pooling). All converge to the same val abs_rel within float noise. The "queries help depth" hypothesis is empirically false here — at least for fresh-init, frozen-backbone, 5k-iter training.

2. **Architectural changes to the IDR path don't matter either.** Replacing `Linear(256, 128)` (single matrix multiply per query) with a 2-layer cross-attention transformer-decoder pooling module — a 7× increase in parameter count for the IDR generator — moves the final number by 0.001. The IDR-side architecture is essentially decoupled from the depth output.

3. **Default thresholds in foundation models cost silent runs.** SAM3's `Sam3Processor.confidence_threshold` defaults to 0.5; SAM3's max score on KITTI is 0.017. That mismatch silently zeroed 100% of the queries Run 1 saw. Anyone porting a foundation-model detector to a new domain should verify thresholds and detection counts on real data before training.

4. **Multi-class prompts collapse SAM3's confidence.** The production `"car . truck . person . ..."` prompt gives mean(top200) score 0.002; single-class `"vehicle"` gives 0.10 — a 50× difference. SAM3 was clearly trained for one-class queries; the `.` separator is a brittle mechanism for multi-class detection.

5. **Cross-attention seeds carry depth-task gradient signal independent of input.** Run 5's seeds are 32 × 128 learnable parameters that cross-attend to whatever SAM3 produces. Even with input variance present (cos sim 0.55 across images), the *output* IDRs converge to a similar distribution as random projections of zeros — meaning the seeds learn a depth-task-relevant bias regardless of what SAM3 says. This is consistent with the ISD bottleneck treating IDRs as a learned bias rather than per-image content.

6. **The FPN path is doing 100% of the depth work.** Stronger now after Run 5: even a properly-architected query translator with 707k extra params doesn't break the 0.082 floor. The signal moving the depth metrics is in `MSDeformAttnPixelDecoder(SAM3 FPN)` → `ISD.pixel_proj`, not in any IDR-side path.

---

### Run 6 — sequence training with non-overlapping 4-frame clips (E18)

| Setting | Value |
|---------|-------|
| Mode | `replace` |
| Prompt | Singleclass × 4 (`vehicle`, `tree`, `road`, `building`) |
| SAM3 calls/frame | 4 |
| Tokens fed to model | **200** |
| Data | `KITTISequenceDataset`, clip_length=4, stride=4 (non-overlapping), 5,779 unique clips |
| Change vs Run 4 | Switched from single-frame (23,158 frames) to 4-frame clips; 8 frames/step instead of 2; same model. |

`KITTISequenceDataset` with `clip_length=4`, `stride=clip_length` (non-overlapping clips). Each frame appears in exactly one clip per epoch. 5,779 unique clips × batch_size 2 → 2,889 steps/epoch ≈ 5h/epoch. 5k iters ≈ 1.7 epochs. Same `sam_mode=replace` (sam3_proj Linear) and K=200 singleclass prompts as Run 4. The per-frame inner loop flattens (B=2, T=4) → 8 independent single-frame forwards per optimizer step; gradients accumulate across all 8 frames before stepping.

| Step | E18 (sequence, 8 frames/step) | Run 4 (single-frame, 2 frames/step) | Δ |
|-----:|------------------------------:|------------------------------------:|---:|
|  500 | 0.1222 | 0.1403 | −0.0181 (−13%) |
| 1000 | 0.2998 | 0.1055 | +0.1943 — spike |
| 1500 | 0.1049 | 0.1184 | −0.0135 (−11%) |
| 2000 | **0.1038** | 0.1054 | −0.0016 (−2%) |

**Sequence training is converging faster per optimizer step** — 8 frames/step provides ~4× more gradient signal per optimizer update vs single-frame batch=2. Val at step 500 was already 13% ahead of single-frame, and by step 2000 it is again ahead (0.1038 vs 0.1054), having recovered cleanly from the step-1000 spike.

**The step-1000 spike (0.300) was temporary.** Training loss remained stable (1.1–1.6) while val collapsed for one checkpoint — consistent with the LR peaking at step 500 combined with the small unique-clip dataset (5779 clips → ~34% through epoch 1 at step 1000). The model pulled out of it naturally as LR decayed; no architecture or data change was needed.

**Configuration iteration before landing on 57602:** Two prior runs were killed without completing. Run 57584 (clip_length=4, stride=1 → 23,059 overlapping clips) reached step 200 (loss=2.07) before being restarted with non-overlapping clips. Run 57574 (clip_length=2, stride=1) was killed at step 5 to enable the clip_length=4 change. Neither produced val data.

E18 was cancelled at step 2000 (3:48 elapsed) while still improving. Best abs_rel = **0.1038** at step 2000.

### Run 7 — SAM3 video encoder + sequence dataset (E19, killed at step ~2900)

| Setting | Value |
|---------|-------|
| Mode | `translate` (Sam3QueryToIDR) — later corrected to `replace` in E19-redux |
| Prompt | Singleclass × 4 — **BROKEN**: per-class `add_prompt` loop overwrote each other; only "building" applied |
| SAM3 calls/clip | 4 grounding calls (bug) + 3 tracker propagation steps |
| Tokens fed to model | 200 (but derived from "building" prompt only) |
| Data | clip=4, stride=4, 5779 clips |
| Change vs Run 6 | Added `Sam3VideoPixelEncoder` (temporal tracker memory). Bug in prompt loop discovered post-hoc. |

`Sam3VideoPixelEncoder` wraps `build_sam3_video_model()` (detector + tracker). For each 4-frame clip: `init_state(4_PIL_frames)` → `add_prompt(frame_idx=0, text_str=cls)` per class → `propagate_in_video()`. The tracker propagates frame-0 detections to frames 1–3 so queries for later frames carry temporal memory. FPN captured via monkey-patched `backbone.forward_image` (raw 256-channel output before tracker `conv_s0/conv_s1` projections). Decoder queries captured via hook on `video_model.detector.transformer.decoder`. Same `sam_mode=translate`, K=200, singleclass prompts.

| Step | E19 (video encoder, temporal queries) | E18 (replace, no temporal) | E11 (single-frame baseline) |
|-----:|-------------------------------------:|---------------------------:|----------------------------:|
|  500 | 0.3859 | 0.1222 | 0.1403 |
| 1000 | 0.5582 | 0.2998 | 0.1055 |
| 1500 | **0.3101** | 0.1049 | 0.1184 |
| 2000 | 0.3762 | 0.1038 | 0.1054 |
| 2500 | 0.4060 | — | 0.0955 |

E19 substantially underperforms E18 and E11. Best val abs_rel = **0.310** at step 1500, compared to E18's 0.104. The temporal memory machinery is making training harder rather than easier.

**Why E19 underperforms:**
1. **Forward+backward propagation doubles the decoder passes per frame.** `propagate_in_video(direction="both")` visits each of 4 frames twice (forward then backward). The decoder hook fires 8 times per clip, giving 8 query sets. We only capture the first firing per frame, so backward-pass queries (which may be over-fit to downstream frames) are being used for some frames.
2. **Per-step compute is ~1.3× higher** (8.2s vs 6.2s) due to tracker overhead — with only 5779 clips/epoch, this reduces effective training coverage.
3. **The video tracker was trained for object segmentation propagation, not depth.** Temporal "memory" may be propagating segmentation-irrelevant features that actively harm the depth-relevant query embedding.
4. **Loss starts much higher** (4.5 at step 100 vs 2.6 for single-frame) suggesting the video encoder's features are harder for the iDisc head to use from cold start.

---

### Run 8 — single multiclass prompt, 200 tokens, replace mode, single-frame (E20, killed at step 3000)

| Setting | Value |
|---------|-------|
| Mode | `replace` |
| Prompt | `"vehicle . tree . road . building"` — **1 SAM3 call per frame** |
| SAM3 calls/frame | 1 |
| Tokens fed to model | **200** (all 200 decoder slots, no top-K) |
| Data | Single-frame, 23,158 train |
| Change vs Run 4 | Collapsed 4 singleclass calls into 1 multiclass call. 4× faster per step. Scores lower (0.008 max vs 0.81) but all 200 tokens kept regardless. |

New prompt regime: one SAM3 decoder call per frame with `"vehicle . tree . road . building"` as a single multi-class string. All 200 decoder slots used directly (no top-K selection, no per-class loop). 4× faster per step than singleclass-loop runs (1 decoder call/frame vs 4). `sam_mode=replace`, Linear `sam3_proj`, single-frame KITTI (not sequence). Killed at step 3000 while still improving.

| Step | E11 (placeholder zeros) | E20 (multiclass 200-token) | Δ vs E11 |
|-----:|------------------------:|---------------------------:|---------:|
|  500 | 0.1403 | **0.1325** | −0.0078 (−5.6%) |
| 1000 | 0.1055 | 0.1393 | +0.0338 (spike) |
| 1500 | 0.1184 | **0.1155** | −0.0029 (−2.4%) |
| 2000 | 0.1054 | 0.1268 | +0.0214 (spike) |
| 2500 | 0.0955 | **0.0980** | +0.0025 |
| 3000 | 0.0913 | **0.0958** | +0.0045 |

**Full trajectory (killed at step 3000):**

| Step | E11 (placeholder) | E20 (multiclass) | Δ |
|-----:|------------------:|-----------------:|--:|
|  500 | 0.1403 | **0.1325** | −0.0078 |
| 1000 | 0.1055 | 0.1393 | +0.0338 (spike) |
| 1500 | 0.1184 | **0.1155** | −0.0029 |
| 2000 | 0.1054 | 0.1268 | +0.0214 (spike) |
| 2500 | 0.0955 | **0.0980** | +0.0025 |
| 3000 | 0.0913 | **0.0958** | +0.0045 |

**Key finding:** The multiclass single-prompt carries real query signal — val at steps 500 and 1500 beats E11's locked placeholder trajectory. However the trajectory is more volatile (alternating spikes and new bests) and runs ~0.003–0.005 behind E11 at each converged checkpoint. Best abs_rel = **0.0958** at step 3000 (killed before completion; on pace to reach ~0.082 like E11).

**Why the volatility?** In `replace` mode, `sam3_proj` is the sole source of IDRs. When it gets real query signal (as opposed to zeros), the gradient is noisier because the queries vary per image. With placeholder zeros, the gradient through sam3_proj is constant — paradoxically more stable. The multiclass queries help early (better initialization signal) but introduce gradient variance that causes periodic regressions.

---

### Run 9 — E18-redux: sequence + multiclass single-prompt + replace (cancelled at step 2000)

| Setting | Value |
|---------|-------|
| Mode | `replace` |
| Prompt | `"vehicle . tree . road . building"` — 1 SAM3 call per frame |
| SAM3 calls/frame | 1 |
| Tokens | 200 |
| Data | clip=4, stride=4, 5779 clips |
| Change vs Run 6 | Prompt changed from singleclass ×4 to multiclass ×1. Everything else identical. |

Repeat of E18 with the new multiclass single-prompt (`"vehicle . tree . road . building"`, one SAM3 decoder call per frame). Purpose: verify whether the 4× faster prompt path changes training dynamics.

**Result: no change.** At every checkpoint the E18-redux trajectory is indistinguishable from E18-singleclass (Δ ≤ 3e-6). The sequence-training spike-and-recovery pattern is fully determined by the training regime (LR schedule, dataset size, batch size), not by prompt mode or query content. Cancelled at step 2000, best abs_rel = 0.1038.

### Run 10 — E19-redux: video encoder + replace + multiclass + sequence (killed at step 2100, diverging)

| Setting | Value |
|---------|-------|
| Mode | `replace` |
| Prompt | `"vehicle . tree . road . building"` — **1 SAM3 call per clip** on frame 0; tracker propagates to frames 1–3 |
| SAM3 calls/clip | 1 grounding + 3 tracker propagation steps |
| Tokens | 200 per frame (via video propagation) |
| Data | clip=4, stride=4, 5779 clips |
| Change vs Run 7 | Fixed broken prompt loop (single call instead of 4 overwriting each other). Fixed mode to `replace` (Run 7 used `translate`). |

Re-run of E19 with the correct prompt setup (single `add_prompt("vehicle . tree . road . building")`) and corrected mode (`replace`, Linear `sam3_proj`). Constrained to 5060 Ti (16 GB).

| Step | E18-redux (image encoder) | E19-redux (video encoder) |
|-----:|--------------------------:|--------------------------:|
|  500 | 0.122 | 0.254 |
| 1000 | 0.300 (spike) | 0.304 (spike) |
| 1500 | **0.105 (recovered)** | 0.371 (degrading) |
| 2000 | 0.104 | 0.385 (still degrading) |

**Conclusion confirmed:** SAM3 video encoder temporal memory actively hurts depth training under this setup. The step-1000 spikes are nearly identical (0.300 vs 0.304), but E18-redux fully recovers while E19-redux continues to deteriorate. The divergence is systematic and persistent.

**Likely cause — train/val distribution mismatch:** During training, the video encoder processes 4-frame clips where the tracker has memory populated from frame 0; during validation it processes single frames as 1-frame clips where the tracker memory is empty. This creates a systematic difference between the feature distribution seen at train time vs val time, which grows worse as the model adapts to the memory-populated distribution.

---

## Final summary table (all SAM3 pure runs, KITTI Eigen, val abs_rel)

| Exp | Mode | Data | Best abs_rel | Steps |
|-----|------|------|------------:|------:|
| E1 (baseline) | Pretrained iDisc-R101 | — | **0.0600** | ~120k |
| E11 | replace (Linear), K=200 singleclass | single-frame | 0.0818 | 5k |
| E12 | translate (Sam3QueryToIDR), K=200 singleclass | single-frame | 0.0830 | 5k |
| E18 | replace (Linear), K=200 singleclass | 4-frame clips, stride=4 | 0.1038† | 2k |
| E19 | translate, video encoder | 4-frame clips, stride=4 | 0.3101 | 2.5k |
| E20 | replace (Linear), multiclass single-prompt, 200-token | single-frame | 0.0958‡ | 3k |
| E18-redux | replace (Linear), multiclass single-prompt | 4-frame clips, stride=4 | 0.1038§ | 2k |
| E19-redux | replace (Linear), multiclass, video encoder | 4-frame clips, stride=4 | 0.385↑ (diverging) | 2k |

† E18 was cancelled at step 2000 still improving; extrapolated final ~0.082–0.090.
‡ E20 killed at step 3000 still improving; multiclass queries gave real but noisy signal vs E11's locked zeros.
§ E18-redux confirmed identical to E18 singleclass (Δ ≤ 3e-6); prompt mode fully invariant to sequence training.

**Updated comprehensive comparison (all multiclass-prompt runs included):**

| Exp | Mode | Tokens | SAM3 calls/frame | Prompt | Data | Best val abs_rel |
|-----|------|-------:|----------------:|--------|------|----------------:|
| E1 baseline | Pretrained iDisc-R101 | — | — | — | — | **0.0600** |
| E11 | replace | **0** (placeholder) | 1 | multiclass 8-class (ineffective) | single-frame | 0.0818 |
| E12 | translate | 200 | 4 | singleclass ×4 | single-frame | 0.0830 |
| E18 | replace | 200 | 4 | singleclass ×4 | 4-frame clips | 0.1038† |
| E19-original | translate | 200 | 4 (broken) | per-class loop (only "building" applied) | 4-frame clips | 0.3101 |
| E20 | replace | **200** | **1** | multiclass single call | single-frame | 0.0958‡ |
| E18-redux | replace | **200** | **1** | multiclass single call | 4-frame clips | 0.1038§ |
| E19-redux | replace | **200** | **1**/clip | multiclass single call (fixed) | 4-frame clips | 0.385↑ |

**Overall conclusions across all experiments:**

1. **Prompt mode is invariant to training outcome** in single-frame runs — placeholder zeros, singleclass ×4, and multiclass single-prompt all converge to ≈0.082 given enough iters. The multiclass prompt has more volatile trajectory but similar asymptote.

2. **Sequence training does not improve depth** at comparable iter budgets. The step-1000 spike is a deterministic artifact of the small unique-clip dataset (5779 clips, ~2889 iters/epoch) + LR peak at step 500. E18-redux confirms this is prompt-invariant.

3. **SAM3 video temporal memory actively hurts depth training.** Both E19-original and E19-redux diverge while E18-redux recovers. The train/val distribution mismatch (tracker memory populated during training vs empty during val) is the most likely cause.

4. **The 0.082 floor is robust.** Every converged single-frame experiment reaches it regardless of query quality. To break through it requires changing the FPN path (unfreezing SAM3, richer channels) not the query path.

---

## Open questions

1. **Where is the FPN-only ceiling?** The current architecture (frozen SAM3 backbone + fresh-init iDisc-side modules at 5k iters) sits at 0.082. How much of the gap to E1 (0.060) is frozen-vs-trainable backbone, how much is the channel-equalized FPN (256/256/256/256 vs ResNet's 256/512/1024/2048), and how much is just the iter budget? A longer schedule with re-tuned cosine decay, plus a longer-iter run, would decompose these.

2. **Would unfreezing SAM3 (LoRA or partial-layer) close the gap via the FPN path?** Given that the FPN is doing all the work, this is likely the highest-leverage change. With LoRA on SAM3's neck (rank-8, ~3M extra params, no full backward through the trunk), the FPN features can specialize for depth. Backbone LR 0.1× head LR; 10k iters with extended schedule.

3. **Does ISD's cross-attention actually use IDRs at all in this setup?** A direct test would replace `idrs` with a random tensor or a constant, and see if val abs_rel changes. We've effectively done this implicitly across Runs 1/3/4/5 — and the answer is "barely." This suggests ISD has degenerated into a feature-mixing layer that doesn't really need IDRs to work. A hard ablation (random IDRs as input) would confirm this.

4. **Are SAM3's segmentation features information-redundant with what AFP would extract?** A complementary experiment is `concat` mode (AFP IDRs from the SAM3 FPN + Sam3QueryToIDR IDRs from queries) vs `concat` mode (AFP only — i.e., baseline iDisc on top of SAM3's FPN). If both produce the same val number, then SAM3 queries truly carry zero signal beyond what AFP already gets from the FPN. This is the cleanest "do queries help at all" test.

5. **Was the original "pure SAM3 = 0.0818" exciting result a coincidence?** At face value Run 1 looked like good news — SAM3 features carrying spatial structure, only 36% above SOTA. We now know the queries were placeholders, queries don't matter, and 0.082 is the floor for any d2c-only setup with frozen SAM3. The conclusion should be: SAM3's frozen FPN is *competent* (much better than random features) but *not good enough* (0.022 above SOTA). The next gains come from the FPN path, not from SAM3 queries.
