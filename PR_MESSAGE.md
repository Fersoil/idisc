## Integrate SAM3 instance queries into iDisc depth estimation

**Hypothesis:** SAM3's per-instance query embeddings encode semantic and spatial information about objects. Injecting them into iDisc's IDR slot (instead of, or alongside, the AFP output) should improve depth at object boundaries and occluded regions — the model's known failure cases.

The main model change is `sam3_proj` (3 × Linear(256→128), one per IDR resolution), which projects SAM3's instance queries into IDR space. `IDisc.forward()` gains `instance_queries` and `sam_mode` (`replace` / `concat`) to select the integration path. See [EXPERIMENTS.md](EXPERIMENTS.md) for the full experiment grid.

Two implementation details worth reviewing:
- **Normalisation fix**: SAM3 was receiving double-normalised images; fixed by denormalising before the SAM3 forward pass.
- **Linear projection replaces avg_pool2d**: the earlier s-seq integration used spatial pooling on hidden states; the concat/replace paths now use the learned `sam3_proj` instead.

### Test plan

- [ ] `experiment=baseline` matches known iDisc KITTI metrics
- [ ] `pooled_empty/multiclass/singleclass` produce identical depth numbers (prompt must not affect pooled path)
- [ ] `experiment=concat_singleclass` runs end-to-end without shape errors
- [ ] `experiment=finetune_concat_singleclass finetune=fast` completes one epoch
