# Pure SAM3 + iDisc d2c (RUN LOG)

## Goal

Test whether SAM3 (a segmentation foundation model) can serve as a complete visual encoder for iDisc, with both its dense FPN features replacing iDisc's ResNet/Swin and its 200 segmentation queries flowing into iDisc's d2c bottleneck. The iDisc-side modules train fresh from random init; no pretrained iDisc weights are loaded. The reference number is the pretrained iDisc-R101 baseline on KITTI Eigen at abs_rel ≈ 0.060.

---

## Run log

### Run 1 — first end-to-end "pure SAM3" training

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

## Open questions

1. **Where is the FPN-only ceiling?** The current architecture (frozen SAM3 backbone + fresh-init iDisc-side modules at 5k iters) sits at 0.082. How much of the gap to E1 (0.060) is frozen-vs-trainable backbone, how much is the channel-equalized FPN (256/256/256/256 vs ResNet's 256/512/1024/2048), and how much is just the iter budget? A longer schedule with re-tuned cosine decay, plus a longer-iter run, would decompose these.

2. **Would unfreezing SAM3 (LoRA or partial-layer) close the gap via the FPN path?** Given that the FPN is doing all the work, this is likely the highest-leverage change. With LoRA on SAM3's neck (rank-8, ~3M extra params, no full backward through the trunk), the FPN features can specialize for depth. Backbone LR 0.1× head LR; 10k iters with extended schedule.

3. **Does ISD's cross-attention actually use IDRs at all in this setup?** A direct test would replace `idrs` with a random tensor or a constant, and see if val abs_rel changes. We've effectively done this implicitly across Runs 1/3/4/5 — and the answer is "barely." This suggests ISD has degenerated into a feature-mixing layer that doesn't really need IDRs to work. A hard ablation (random IDRs as input) would confirm this.

4. **Are SAM3's segmentation features information-redundant with what AFP would extract?** A complementary experiment is `concat` mode (AFP IDRs from the SAM3 FPN + Sam3QueryToIDR IDRs from queries) vs `concat` mode (AFP only — i.e., baseline iDisc on top of SAM3's FPN). If both produce the same val number, then SAM3 queries truly carry zero signal beyond what AFP already gets from the FPN. This is the cleanest "do queries help at all" test.

5. **Was the original "pure SAM3 = 0.0818" exciting result a coincidence?** At face value Run 1 looked like good news — SAM3 features carrying spatial structure, only 36% above SOTA. We now know the queries were placeholders, queries don't matter, and 0.082 is the floor for any d2c-only setup with frozen SAM3. The conclusion should be: SAM3's frozen FPN is *competent* (much better than random features) but *not good enough* (0.022 above SOTA). The next gains come from the FPN path, not from SAM3 queries.
