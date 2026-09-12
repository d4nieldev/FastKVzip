# LongBench v2 — implementation decisions

These choices fill details left open by the [approved plan](plan.md).
Legend: 🟠 **Plan gap** means an implementation choice not separately approved
by the user. It does not mean a departure from the agreed objective.

## D1 — 🟠 Plan gap — Define the exact 1M capacity and smaller-model behavior

- **Background:** The plan specifies “1M capacity.” Qwen's model configuration permits 1,010,000 positions, while other supported models may permit fewer.
- **Decision:** Interpret 1M as exactly 1,000,000 total tokens, including the reserved prompt and scoring/generation space. Use the lower of that budget and the loaded model's declared position capacity. Fail before prefill when the capacity is unknown or the protected prompt cannot fit.
- **Plan gap or deviation:** The plan did not define the exact integer behind “1M” or the behavior for a model with a smaller capacity.
- **Reason and tradeoff:** This keeps the comparison budget fixed without claiming that every model can handle one million positions. It does not exploit Qwen's additional 10,000 positions or guarantee that the full KV cache fits GPU memory.
- **Status:** 🟠 Plan gap; implemented without separate user approval of these numerical/error-handling details.

| Code reference | What this code does |
| --- | --- |
| [Budget constants](https://github.com/d4nieldev/FastKVzip/blob/898421c091b5ea6709a08d47f4e0a1f71611927d/prefill/data/longbench_v2.py#L7-L10) | Defines the exact one-million-token budget and 128-token answer cap. |
| [Capacity validation and reservation](https://github.com/d4nieldev/FastKVzip/blob/898421c091b5ea6709a08d47f4e0a1f71611927d/prefill/data/wrapper.py#L72-L91) | Limits the budget to the model capacity and reserves the larger answer or reconstruction tail. |

## D2 — 🟠 Plan gap — Preserve a shared context when checkpoint prefixes differ

- **Background:** GraphKV restores its saved prefix after dataset initialization. Baselines use the native model prefix. Subtracting each actual prefix length independently would give different context slices when their lengths differ.
- **Decision:** Calculate the common context budget from the native model prefix. Keep the exact GraphKV prefix, but reject an example if a longer saved prefix cannot fit after the shared context is selected. Shorter prefixes do not grant GraphKV additional context.
- **Plan gap or deviation:** The plan requires identical context across methods and preservation of the checkpoint prefix. It did not specify how to reconcile those requirements when a custom prefix consumes more space.
- **Reason and tradeoff:** Rejecting the incompatible case preserves the comparison instead of shortening only GraphKV's context or altering its saved prefix. Custom longer prefixes can still evaluate examples that fit without exceeding the budget.
- **Status:** 🟠 Plan gap; implemented without separate approval of the incompatible-prefix error behavior.

| Code reference | What this code does |
| --- | --- |
| [Native prefix capture](https://github.com/d4nieldev/FastKVzip/blob/898421c091b5ea6709a08d47f4e0a1f71611927d/prefill/data/wrapper.py#L39-L41) | Remembers the native prefix length before checkpoint restoration. |
| [Shared context and guard](https://github.com/d4nieldev/FastKVzip/blob/898421c091b5ea6709a08d47f4e0a1f71611927d/prefill/data/wrapper.py#L86-L99) | Selects the same context budget and checks that the actual protected prefix still fits. |
| [Different-prefix regressions](https://github.com/d4nieldev/FastKVzip/blob/898421c091b5ea6709a08d47f4e0a1f71611927d/prefill/tests/test_longbench_v2_runtime.py#L70-L87) | Tests identical slices with shorter prefixes and rejection of incompatible longer prefixes. |

## D3 — 🟠 Plan gap — Expose the published evaluation file as logical test data

- **Background:** Hugging Face labels LongBench v2's 503 evaluation questions as a `train` split. Our evaluation loader defaults to `test`.
- **Decision:** Load the pinned `data.json` through the existing logical `test` path. Reject other split names for this benchmark.
- **Plan gap or deviation:** The plan names the official dataset but does not specify how its upstream split name maps to our existing loader contract.
- **Reason and tradeoff:** All evaluation entry points work without a new split argument or an evaluator-specific branch. The upstream name remains documented; it does not imply training on these benchmark answers.
- **Status:** 🟠 Plan gap; implemented without separate approval of the split alias.

| Code reference | What this code does |
| --- | --- |
| [Canonical loader branch](https://github.com/d4nieldev/FastKVzip/blob/898421c091b5ea6709a08d47f4e0a1f71611927d/prefill/data/load.py#L77-L82) | Accepts logical test and delegates to the pinned concrete loader. |
| [Pinned file loader](https://github.com/d4nieldev/FastKVzip/blob/898421c091b5ea6709a08d47f4e0a1f71611927d/prefill/data/longbench_v2.py#L40-L56) | Loads the full evaluation file while preserving range order and full benchmark size. |
