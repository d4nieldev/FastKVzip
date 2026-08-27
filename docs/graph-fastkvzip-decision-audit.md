# Implicit Whole-Context Graph FastKVzip: Decision Audit

This is the review guide for the implicit mixer implementation in this
worktree. It describes current code, not a future design.

Each row states the decision, why it exists, alternatives, and its strongest
conversation source. Source order is: user formulation, user agreement,
approved plan, inherited behavior, then implementation-only.

## Complete pipeline

For one dataset context:

1. The teacher pre-fills prefix plus context in LLM prefill chunks.
2. The retain cache captures one CPU hidden-state tensor per transformer layer.
3. KVzip reconstruction produces teacher labels for context positions.
4. The trainer releases the teacher KV cache and keeps CPU hidden tensors,
   labels, token IDs, prefix IDs, and context identity.
5. Each layer/KV-head graph makes a whole-context implicit mixer result.
6. The matching FastKVzip gate scores the original hidden state plus that
   residual.
7. Joint mode performs one gate and one mixer update per context. Two-phase
   mode performs gate token-slice updates followed by one mixer update.
8. Evaluation restores the exact prefix and prefill chunk, scores only
   context positions, applies the existing local-window protection, clears
   hidden states, then reuses the existing prune/generate loop.

Main source files:

- [implicit mixer and gate adapter](../prefill/graph/model.py)
- [streamed training and checkpoints](../prefill/graph/training.py)
- [checkpoint reconstruction and evaluation scoring](../prefill/graph/evaluation.py)
- [CLI, teacher generation, cache, and W&B](../prefill/train_graph.py)
- [standalone evaluation entry point](../prefill/eval_graph.py)

## Terms and shapes

| Name | Shape | Meaning |
|---|---:|---|
| P | scalar | prefix/system/start-of-turn token count |
| T | scalar | scored context token count |
| L, H | scalar | transformer layers and KV heads |
| D, C | scalar | hidden size and graph dimension |
| M | scalar | complete graphs in a graph microbatch |
| X | conceptually [M,T,D] | selected layer/head hidden states |
| Y1, Y2 | [M,T,C] | low-rank projections |
| S | [M,C,C] | Gram contraction |
| scores | [L,1,H,T] | cache importance scores |

The [M,T,D] form is mathematical. Production retains Y1[M,T,C], small
contractions/statistics, and one token slice of hidden-width work. It never
retains a full mixer output or a T-by-T adjacency.

## Scope and integration

| Decision | Why | Alternatives | Source |
|---|---|---|---|
| Use an implicit low-rank mixer instead of a materialized token graph. | Removes expensive neighbor search and avoids a T-by-T adjacency. | Hard k-NN graph or dense adjacency. | User formulation: “we don't actually need to materialize the adjacency matrix.” |
| Keep the FastKVzip gate unchanged. | Isolates the new experiment and preserves released/local gate checkpoints. | Replace the gate. | Approved plan: keep the FastKVzip gate. |
| Keep eval_graph.py standalone and leave eval.py unchanged. | Baseline evaluation stays easy to review. | Add branches to eval.py. | User formulation: “What if we have a brand new script?” |
| Keep the existing dataset split/cursor. | Runs and resumes stay comparable. | New data runner. | Approved plan. |
| Keep production dependencies limited to the implicit mixer path. | The implementation needs no topology-builder runtime. | Carry an unused graph-library dependency. | Approved plan. |

## Teacher context, prefix, and labels

| Decision | Why | Alternatives | Source |
|---|---|---|---|
| Teacher loads with gate_path_or_name empty. | Labels are KVzip reconstruction labels, not a released-gate score. | Teacher with fastkvzip gate. | Approved plan. |
| Score/train context positions only. | Prefix/system/turn tokens must stay protected. | Score all prefill tokens. | User agreement that train/eval hidden states must match. |
| Store exact prefix IDs and prefill chunk in checkpoints. | Both can affect hidden states. | Rebuild a fresh prefix or choose an eval chunk. | User formulation: “if we change the system prompt or prefix in evaluation things might not work the same.” |
| Capture ordinary CPU hidden tensors outside inference mode. | Mixer parameter gradients need normal tensor inputs. | Clone later or retain GPU activations. | Implementation-only: PyTorch inference-tensor requirement. |
| Transfer captured hidden storage into TeacherExample without a second full clone. | Avoids doubling the largest CPU tensor. | Clone all hidden states. | Approved plan: transfer ownership. |
| Normalize scores, token IDs, and prefix IDs to CPU. | They are small and make cache data device-independent. | Keep them on teacher GPU. | Implementation-only. |

The extraction function slices kv.start_idx:kv.end_idx. Prefix data can remain in
the backing CPU tensor, but it is not a training target.

## Implicit mixer architecture

For flattened graph ID g = layer times H plus head:

    Y1 = X W1[g]
    Y2 = X W2[g]
    S  = Y1 transpose Y2 / T       # or unscaled
    R  = LeakyReLU(Y1 S W[g])
    f  = ContextBatchNorm_g(R)
    X' = X + alpha[g] * (gamma[g] * f + beta[g])
    score = matching FastKVzip gate head(X')

| Decision | Why | Alternatives | Source |
|---|---|---|---|
| Every layer/KV head owns W1, W2, W, gamma, beta, and alpha. | Each head can learn a different context relation. | Share by head or layer. | User formulation: “we should have different weights for every head.” |
| Use the corrected Y1(Y1 transpose Y2)W formula. | Final W maps C back to D. | Earlier incompatible annotation. | User formulation: “a different W there that is c x d.” |
| Pack W1/W2 into one D-to-2C projection. | One batched projection is faster; slices stay independent. | Two projection modules. | Approved plan. |
| Use bias-free Kaiming-uniform W1/W2/W. | Exact requested parameterization. | Bias, Xavier, or zero initialization. | Approved plan. |
| Default graph dimension is 32. | Requested latent default. | Smaller/larger default. | User formulation: graph dim 32. |
| Default Gram normalization is token-count; allow none. | Keeps scale stable as T changes. | Always unscaled. | Approved plan. |
| Use LeakyReLU slope 0.01. | Negative messages survive. | ReLU, GELU, learned slope. | Approved plan. |
| Initialize gamma=1, beta=0, alpha=0.1; alpha is unconstrained. | Residual starts small and can learn either sign. | Zero, sigmoid-constrained, or per-token alpha. | Approved plan. |
| Keep signed weights and implicit self-connections. | Y1Y1-transpose may be signed and diagonal entries are self messages. | Clamp, remove diagonal, normalize rows. | Approved plan. |

## Context BatchNorm

| Decision | Why | Alternatives | Source |
|---|---|---|---|
| Compute independent stats per layer/head across that context's T tokens. | No layer/head mixes statistics. | Batch stats across graphs. | Approved plan. |
| Use population variance, divide by T. | Explicit requested behavior. | Divide by T-1. | Approved plan. |
| Add 1e-5 inside square root. | Singleton/constant contexts remain finite. | No epsilon or another epsilon. | Approved plan. |
| Keep no running averages; recompute at train and eval. | Current context must define normalization. | Standard BatchNorm running state. | Approved plan. |
| Use learnable per-feature gamma/beta. | Matches the formula. | No affine transform. | Approved plan. |
| Accumulate Gram/statistics in FP32, but preserve FP64 in numerical tests. | Stable half/bfloat16 production and exact float64 verification. | Always model dtype. | Implementation-only. |
| Merge streamed moments with Chan/Welford arithmetic. | Exact population stats without retaining R[M,T,D]. | Materialize all raw messages. | Implementation-only. |

## Chunking, memory, and exact gradients

| Decision | Why | Alternatives | Source |
|---|---|---|---|
| Prefill chunks do not split the graph. | Later LLM chunks attend prior KV cache; the result is one context. | Independently prefill subcontexts. | User agreement: hidden states should be identical in train/eval. |
| Graph microbatch auto equals H. | One complete layer is default. | One graph or all graphs. | Approved plan. |
| Explicit graph microbatch must be 1 through L times H. | Invalid graph batches fail before teacher generation. | Clamp silently. | Approved plan. |
| Token microbatch defaults to 1,000. | Bounds temporary hidden-width work. | Full-context hidden-width ops. | Approved plan. |
| Retain Y1[M,T,C], not raw messages/residuals. | C is much smaller than D. | Retain full R or delta. | Approved plan. |
| Use two streamed loss passes for exact BatchNorm gradient. | BatchNorm couples all T tokens. | Treat token chunks as independent BN batches. | Approved plan. |
| Backpropagate complete graph gradients before an optimizer step. | Mixer gets one update per context, not T/1000 updates. | Update per token chunk. | User question about whether the graph update happens T/1000 times, then approved clarification. |
| Verify staged float64 gradients against ordinary full autograd. | Tests the streamed algebra. | Only check finite loss. | Approved plan. |

The implemented population-BatchNorm backward is:

    dR = invstd * (dZ - mean(dZ) - Z * mean(dZ * Z))

Z is normalized R. Training then maps the full Gram gradient back to Y1/Y2 and
back through the packed input projection.

## Training modes and optimizers

| Decision | Why | Alternatives | Source |
|---|---|---|---|
| Joint mode is default. | One gate and one mixer update per whole context. | Two-phase default. | Approved plan. |
| Joint mode permits different gate/mixer LRs and schedulers. | A pretrained gate can use lower LR than a new mixer. | Require equality/copy settings. | User formulation: “why can't we have separate learning rates … for the mixer and for the gate?” |
| Defaults are gate LR 1e-4 and mixer LR 1e-3. | Mixer starts from scratch. | One shared LR. | Approved plan. |
| Support PyTorch scheduler names plus JSON kwargs. | Reuses standard scheduler behavior. | Custom scheduler language. | Approved plan. |
| Step normal schedulers after their optimizer; plateau after validation. | Matches PyTorch semantics. | Step before optimizer. | Approved plan. |
| Two-phase remains optional. | Gate has ceil(T/1000) shuffled updates; mixer has one context update. | Remove staged mode. | Approved plan. |
| A cadence step means one completed training context. | Joint and two-phase have different numbers of inner optimizer calls. | Count raw optimizer calls or token slices. | User question: “After each step?” |
| Default checkpoint cadence is every training context; default validation cadence is every epoch. | Keeps the existing recovery point while avoiding no-op saves after validation examples. | Save every validation context. | User formulation: “in validation there is no point of saving the model because its validation the model does not change”. |
| Gate phase sees current mixed inputs. | Prevents raw-only gate preference. | Train it on raw hidden states. | User concern that the gate may “prefer the raw hidden states.” |
| AdamW decays W1/W2/W only; alpha/gamma/beta get zero decay. | Projection and affine parameters need different regularization. | One decay group. | Approved plan. |
| Existing gate parameter grouping is unchanged. | Preserves FastKVzip behavior. | Repartition gate weights. | Approved plan. |

The new preferred CLI names say mixer. Existing graph-LR spellings are aliases
for command compatibility. Checkpoints use the unambiguous mixer names.

## Teacher cache and resume

| Decision | Why | Alternatives | Source |
|---|---|---|---|
| Teacher cache directory is optional. | Missing option preserves online-only generation. | Require disk cache. | Approved plan. |
| Cache one file per dataset name/example index. | Partial caches and resume work naturally. | One monolithic file. | Approved plan. |
| Store hidden states, scores, token IDs, prefix IDs, identity, length, model ID, and prefill chunk. | A cache hit can be validated before prefill. | Store activations only. | Approved plan. |
| Check cache before dataset/wrapper/prefill construction. | A hit avoids all teacher work. | Check later. | Implementation-only. |
| Fail on corrupt/incompatible cache. | Never silently train on wrong teacher data. | Regenerate/overwrite automatically. | Approved plan. |
| Publish cache with temp file plus hard link, never replacement. | Atomic final creation without overwriting an existing cache. | os.replace. | Implementation-only. |
| Cache path is not a checkpoint resume invariant. | A resume can use a different scratch path. | Store/compare it in checkpoint config. | Approved plan. |
| Save `last.pt` at the configured training-context or epoch cadence, after any due full validation sweep. | A plateau scheduler's state is included while no checkpoint is written per held-out context. | Save after every validation context or before validation. | User request for save strategy/every controls. |
| Save `best.pt` only after a completed validation sweep improves the mean BCE. | It records the model selected by held-out data. | Save a best checkpoint per held-out context. | Existing best-checkpoint behavior, clarified by the new cadence. |
| Load current checkpoint configuration and state strictly. | Architecture mismatches fail at load time. | Partial or permissive loading. | Approved plan. |

Each checkpoint includes mixer/gate state, both optimizer/scheduler states,
architecture config, model ID, prefix IDs, prefill chunk, cursor, RNG state,
and W&B run ID. State loading is strict.

## Evaluation and protection

| Decision | Why | Alternatives | Source |
|---|---|---|---|
| Restore checkpoint prefix IDs and prefill chunk. | Reproduces training hidden-state conditions. | Use current defaults. | Approved plan. |
| Score only kv.start_idx:kv.end_idx. | Prefix/start-of-turn tokens are not context nodes. | Score entire cache. | Approved plan. |
| Preserve prefix/turn/query/postfix/local-window/generated protection in existing cache code. | Mixer must not redefine cache safety. | New custom pruning code. | Approved plan. |
| Apply existing local-window score override. | Keeps pruning behavior unchanged. | Hard mask or no window. | Approved plan. |
| Always clear hidden cache after score assignment. | It is the largest temporary. | Retain through generation. | Approved plan. |

Evaluation calls the existing DataWrapper, evaluator, ratio loop, prune method,
and result saver from the standalone eval_graph.py entry point.

## Logging and observability

| Decision | Why | Alternatives | Source |
|---|---|---|---|
| Log training metrics once per training context, and `validation/bce` once per full validation sweep. | The validation value matches scheduler and best-checkpoint selection. | Log four partial validation BCE values. | User question: “what evaluation metrics are logged to w&b?” |
| Put all optimization losses and learning rates under `train/`. | The dashboard has one training section instead of separate joint/gate/mixer sections. | One namespace per phase. | User formulation: “instead of having joint/ mixer/ and gate/ sections we can just put everything under train/.” |
| Log `train/bce` in joint mode, and `train/gate_bce` plus `train/mixer_bce` in two-phase mode. | The metric name retains the active phase without creating another W&B section. | Overload one BCE key or use phase namespaces. | Implementation of the user direction above. |
| Put phase timing under `timing/`, not `train/`. | Performance measurement is not an optimization loss. | Put timing under `train/`. | User formulation: “forward and backward times into another namespace it does not belong in train/.” |
| Divide every timing value by the example's scored context length T, and name it `*_seconds_per_token`. | It makes contexts of different lengths comparable. | Log total seconds. | User formulation: “timing metrics to be divided by the context length of the example.” |
| Do not emit `gpu/*` peak-memory metrics. | W&B's asynchronous system monitor already reports GPU state, and duplicate charts clutter the run. | Keep PyTorch peak metrics alongside the system monitor. | User formulation: “we also dont really need the gpu section because system reports this.” |
| Keep mean `validation/bce` under `validation/`. | Held-out loss stays visibly separate from updates. | Put it under `train/` or log individual-context BCEs. | Approved plan, refined by the new cadence. |
| Use W&B's asynchronous system monitor for GPU utilization. | Avoid synchronous utilization polling in the loop. | Poll nvidia-smi each token. | Approved plan. |
| Do not log delta energy. | Explicitly removed from metric set. | Retain old metric. | Approved plan. |

CUDA event timing synchronizes only at the context boundary.

The external metric keys are `train/bce`, `train/gate_bce`,
`train/mixer_bce`, `train/gate_learning_rate`,
`train/mixer_learning_rate`, `validation/bce`, and
`timing/<phase>_{forward,backward}_seconds_per_token`. Internal `graph`
timing is reported as `mixer` because it measures the implicit mixer phase.
`validation/bce` is the mean across the four held-out contexts. Standalone
benchmark evaluation does not log to W&B.

## CLI and operational behavior

| Decision | Why | Alternatives | Source |
|---|---|---|---|
| Keep gate-dim terminology. | It describes FastKVzip dimension directly. | Generic dim. | User formulation: “I would like to have --gate-dim instead of --dim.” |
| Keep gate-sink. | It reconstructs FastKVzip learned base keys. | Infer only from checkpoint. | Inherited behavior. |
| Remove topology-builder and residual-B initialization options. | They no longer describe this model. | Ignore deprecated options. | Approved plan. |
| Accept released fastkvzip and local gate checkpoints. | Mixer can start from released or local gate. | Random-only gate. | Approved plan. |
| Reject freeze-gate without a gate/resume checkpoint. | Freezing a random gate is likely accidental. | Allow silently. | Inherited behavior retained. |

Gate sink is not cache prefix protection. It is the count of learned FastKVzip
base keys in gate.k_base.

## Verification

Focused tests cover:

- implicit multiplication against explicit dense algebra;
- packed W1/W2 independence;
- layer/head BatchNorm isolation, singleton/constant contexts, scale modes, alpha;
- no token-by-token matrix output;
- headwise gate adapter parity;
- graph/token microbatch invariance;
- streamed float64 gradients versus full autograd;
- independent joint optimizer/scheduler settings and AdamW groups;
- compact W&B metric names, per-token timing normalization, and no `gpu/*` metrics;
- save/evaluation cadence, complete validation sweeps, and validation-mean logging;
- teacher-cache creation/reuse/partial/mismatch/corruption;
- current checkpoint save/load;
- context-only evaluation scoring, local window, and hidden-cache release;
- no unused topology-library production imports.

One Qwen3-8B Slurm context completed successfully after the PR was pushed.

## Slurm pilot

| Decision | Why | Alternatives | Source |
|---|---|---|---|
| Push PR before pilot so review starts first. | Cluster time is used on reviewable code. | Submit before push. | User formulation: “after the PR will be ready and pushed.” |
| Run sres immediately before submission. | GPU availability is live state. | Use earlier status. | User formulation: “dont check with sres now.” |
| Prefer rtx_pro_6000:1; choose rtx_6000:1 when live availability is better. | Keeps the preferred GPU while avoiding an unnecessarily long queue. | Hard-code one type. | User formulation: “we can request rtx 6000 if there are more available.” |
| Pilot uses one hour, 60G RAM, 40G scratch. Later cached full run uses 600G scratch after recheck. | Pilot measures actual limits before scale-up. | Full resources immediately. | Approved plan. |
| Teacher cache uses node-local scratch; checkpoints/logs use durable shared storage. | Cache is disposable; checkpoints/logs are not. | Put all data in scratch. | Approved plan. |

Slurm parses SBATCH lines before shell commands. The live GPU choice must be
made with sbatch --gpus=selected-type:1 after inspecting sres, not inside a
static batch script.

## Pilot-resolved details

| Decision | Why | Alternatives | Source |
|---|---|---|---|
| Convert FineWeb's selected NumPy indices to Python integers before dataset lookup. | The cluster's dataset version rejects NumPy scalar indices. | Cast at every lookup or rely on implicit conversion. | Implementation-only: first pilot failure. |
| Finish the W&B run with exit code 1 on any exception and 0 only after the full loop completes. | Failed training no longer appears successful in the dashboard. | Always call `finish()` with its default status. | User formulation: “the w&b run should be marked as failed or something because now it looks like success.” |
| Cast the staged low-precision mixer gradient to the loss working dtype before adding it to the FP32 accumulator. | Half/bfloat16 pilots can complete the exact streamed backward without a dtype-copy failure. | Force all staged buffers to model precision or run the whole path in FP32. | Implementation-only: second pilot failure. |
| Use online W&B when `--wandb-mode online`; offline and disabled modes do not log in. | The supplied API key can sync the pilot and training metrics immediately. | Always use offline mode. | User question: “why not use w&b in online mode? I have an api key”. |

The final one-context Qwen3-8B pilot completed after these fixes. Its role was
to validate the actual cluster path; resource limits for a full run still need
the required live `sres` check.

## Deliberate non-decisions

- Baseline eval.py is unchanged.
- No parameter watching, gradient histograms, heatmaps, or extra ablation passes.
- No k-NN topology or T-by-T adjacency is materialized.
- Teacher-cache files are never automatically deleted or overwritten.
