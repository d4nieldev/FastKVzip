# Findings — things to discuss when Dani is back

Rule: record, do not fix. Nothing in this file has been changed in any repo.

Severity key: **bug** = looks like an implementation error in granola, GPS, or
anything else PR #32 introduced. **limit** = a resource ceiling, expected.
**note** = worth mentioning, not a defect.

## Open

### note — GPS is the SMALLER model at equal width, not the larger one

Measured from the startup line each run prints, Qwen2.5-7B-Instruct-1M,
`--graph-dim 32`, 112 layer/head graphs:

| Mixer | Parameters | Per graph |
|---|---:|---:|
| implicit, `--normalization none` | 38,535,280 | 344,065 |
| implicit, `--normalization granola` | 39,252,080 | 350,465 |
| implicit, `--normalization batchnorm` | 39,338,096 | 351,233 |
| **gps** | **26,858,608** | **239,809** |

GPS is 68% of the implicit mixer's size at the same graph width.

Why: hidden-width projections dominate, and the implicit mixer needs three of
them (the two graph projections and the output projection) while GPS needs two
(in and out). Everything the GPS stack adds -- attention, feed-forward, the
layer norms -- lives at graph width 32 and is negligible beside a 3584-wide
projection.

**This reverses what I told Dani when the comparison was designed.** I said GPS
was the larger model and that matching widths might flatter it. The opposite
holds: at `--graph-dim 32` a GPS win cannot be explained by extra capacity.
Worth re-deciding the Stage C comparison on this basis.

Also: batchnorm costs 802,816 parameters over `none`, exactly 2 x 3584 x 112,
which is the hidden-width scale and bias per graph. GraNoLa costs 716,800, or
6,400 per graph, all at graph width. Both match their designs.

### limit — implicit + batchnorm training OOMs at graph microbatch 56

Job 21450087, `rtx_pro_6000` (94.97 GiB usable). Died after 24 seconds in
`_GateAdapter.forward_batch` at `model.py:1753`, trying to allocate 384 MiB
with 254 MiB free. Scorer width 448.

Clean allocation failure, no defect. Ceiling for this arm is between width 224
(graph microbatch 28, completed) and width 448.

## Resolved by observation

_(none yet)_
