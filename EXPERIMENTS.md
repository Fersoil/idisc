# Experiments

KITTI Eigen split, pretrained iDisc ResNet-101. Run: `sbatch scripts/experiments/run_experiment.sh <ID>`

---

## Architecture

```
Image (3, 352, 1216)
  -> pixel_encoder  (ResNet-101, 44.5M)         FROZEN
  -> pixel_decoder  (MSDeformAttn FPN, 9.1M)    FROZEN
  -> AFP            (0.7M)                       FROZEN
  |    32 latents x 128-dim per resolution, 2 iters cross-attn from decoder
  |    output = IDRs (1, 32, 128) x 3 res
  -> sam3_proj      (0.1M) = 3 x Linear(256->128)   NEW, trainable in F exps
  |    projects SAM3 queries into IDR space
  |    output = (1, N, 128) x 3 res
  -> ISD            (4.5M)                       trainable in F exps
  |    3 heads x 2 depth cross-attn (pixels attend to IDRs)
  -> Depth (1, 1, 352, 1216)

Total = 59.0M. Fine-tuning trains sam3_proj + ISD = 4.6M (7.9%)
```

### SAM3 (frozen, external)

- **Image processor**: SAM3 on single image + text prompt -> top-K queries (256-dim) by detection score. Used in E2-E9, F1-F3.
- **Video predictor**: SAM3 with tracking across drive sequence -> all 200 queries/frame by L2 norm, saved as .pt. Used via cache in E10, F4.

### Integration modes

| Mode | ISD sees | AFP? | sam3_proj? |
|------|----------|------|------------|
| baseline | AFP IDRs (32, 128) x3 | y | n |
| branch | avg_pool2d(mean(hs), (32,128)) x3 | n | n |
| replace | sam3_proj(q) -> (N, 128) x3 | n | y |
| concat | cat(AFP, sam3_proj(q)) -> (32+N, 128) x3 | y | y |

branch = old s-seq code, avg_pool2d destroys info, same tensor cloned x3.

### Prompts

| Mode | Prompt | presence |
|------|--------|----------|
| multiclass | `"car . truck . person . bicycle . building . tree . road sign . pole"` | 0.064 |
| singleclass | run 8x, one class each, merge top-32 | 0.84 |
| classonly | = multiclass but no presence score | - |

singleclass >> multiclass for detection quality.

### Video cache (C1)

61 seqs, 23,855 frames. All 200 queries saved per frame (float16). Dataloader picks top-32 by L2 norm (`sam3_top_k`).

---

## 1 Detection only (no depth)

| Prompt | class_prob | presence | combined | det > 0.5 |
|--------|-----------|----------|----------|-----------|
| multiclass | 0.56 | 0.064 | 0.04 | 0.0/img |
| singleclass x8 | 0.71 | 0.84 | 0.60 | 2.7/img |

IDs: `D1-no-prompt`, `D2-singleclass`, `D3-multiclass`, `D4-classonly`

---

## 2 Depth eval, no training

Everything frozen, sam3_proj = random init.

| ID | Mode | Prompt | abs_rel ↓ | rmse ↓ | d1 ↑ |
|----|------|--------|-----------|--------|------|
| E1 | baseline | - | **0.0600** | **2.363** | **0.964** |
| E2 | branch | empty | 0.0662 | 2.447 | 0.960 |
| E3 | branch | multiclass | 0.0666 | 2.457 | 0.959 |
| E4 | branch | singleclass | 0.0660 | 2.436 | 0.960 |
| E5 | replace | multiclass | 0.0663 | 2.402 | 0.962 |
| E6 | replace | singleclass | 0.0699 | 2.394 | 0.962 |
| E7 | concat | multiclass | 0.0602 | 2.364 | 0.964 |
| E8 | concat | singleclass | 0.0604 | 2.363 | 0.964 |
| E9 | concat | classonly | 0.0603 | 2.363 | 0.964 |
| E10 | concat | cached video | 0.0608 | 2.364 | 0.964 |

- branch = ~10% worse
- replace = worse, random proj + untrained ISD
- concat = ~baseline, AFP helps, random SAM3 tokens ignored

---

## 3 Fine-tuning

Frozen: pixel_encoder + pixel_decoder + AFP. Trainable: sam3_proj + ISD = 4.6M (7.9%).
AdamW lr=5e-5, OneCycleLR, 5000 iters, batch 2, val/500 steps.

| ID | Mode | Queries | abs_rel ↓ | rmse ↓ | d1 ↑ | Step | vs E1 |
|----|------|---------|-----------|--------|------|------|-------|
| F1 | replace | online multiclass | 0.0600 | 2.430 | 0.962 | 2000 | +0.1% |
| F2 | replace | online singleclass | 0.0599 | 2.420 | 0.962 | 2500 | -0.1% |
| F3 | concat | online singleclass | **0.0591** | 2.432 | **0.963** | 2000 | **-1.5%** |
| F4 | concat | cached video | 0.0593 | 2.436 | 0.962 | 4000 | -1.2% |

- replace recovers to baseline but doesn't beat it
- concat improves: AFP = stable base, SAM3 = extra signal. F3 best at -1.5%
- F3 > F4: online singleclass slightly better than cached video
- rmse trade-off: all F exps ~2.43 vs 2.36 baseline (better rel error, worse abs on far objects)

---

## 4 Eval F4 checkpoint

Does fine-tuning ISD break AFP pathway?

| ID | What | abs_rel ↓ | rmse ↓ | d1 ↑ |
|----|------|-----------|--------|------|
| E1-ft | AFP only | 0.0598 | 2.415 | 0.962 |
| E10-ft | AFP + cached video | 0.0593 | 2.436 | 0.962 |

- No regression: E1-ft (0.0598) ≈ E1 (0.0600)
- E10-ft (0.0593) < E10 (0.0608) -> fine-tuning taught ISD to use SAM3 features

---

## Run order

```
D1->D2->D3->D4       detection, no deps
E1                    baseline
E2->E3->E4            branch
E5->E6                replace
E7->E8->E9            concat
C1-cache-video        ~4h
E10                   needs C1
F1->F2->F3            fine-tune online SAM
F4                    needs C1
E1-ft                 needs F4
E10-ft                needs F4+C1
```

---

## Files

| Path | What |
|------|------|
| `scripts/experiments/run_experiment.sh` | SLURM dispatcher |
| `scripts/experiments/eval_depth.py` | depth eval (E exps) |
| `scripts/experiments/eval_sam.py` | detection eval (D exps) |
| `scripts/experiments/finetune_sam.py` | fine-tune sam3_proj+ISD (F exps) |
| `scripts/data/cache_sam3_video.py` | cache video queries (C1) |
| `idisc/models/idisc.py` | main model, forward() w/ instance_queries, sam_mode |
| `idisc/models/id_module.py` | AFP + ISD |
| `idisc/dataloders/kitti.py` | dataloader w/ sam3_cache_dir |

## Changes from original iDisc

1. sam3_proj = 3 x Linear(256->128) to project SAM3 queries into IDR space
2. forward() accepts instance_queries + raw_idrs + sam_mode for replace/concat/branch
3. KITTIDataset: sam3_cache_dir + sam3_top_k, loads .pt files, picks top-K by L2 norm
4. Sam3Processor: exposed instance_queries + topk_scores
5. Added denormalization before SAM3 (was double-normalizing w/ ImageNet stats)
6. Replaced avg_pool2d from s-seq branch w/ linear projection
