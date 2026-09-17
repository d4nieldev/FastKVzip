# Implementation decisions — GPS mixer architecture

Choices made while implementing [the approved plan](plan.md) that the plan left
open or that departed from it.

| Marker and label | Meaning |
| --- | --- |
| 🔴 Plan deviation | Contradicts the approved plan without separate approval. |
| 🟠 Plan gap | The plan left this code choice open. |
| 🟢 User-approved amendment | Approved separately, after the plan. |
| ⚪ Superseded | Replaced by a later decision. |

## D1 — 🟠 Plan gap — the GPS block normalizes per token, not over the context

- **Background:**
  - The plan says each GPS branch gets "its own residual connection and normalization" but does not say which kind.
  - The existing mixer normalizes over the whole context, which needs a separate pass to collect statistics before any token can be scored.
- **Decision:** GPS normalizes each token across its own features, with a learned scale and shift per graph.
- **Plan gap or deviation:** The plan named the normalization's position in the block but not its type.
- **Reason and tradeoff:**
  - Per-token normalization needs no statistics pass, so a block can be stacked without adding a pass per block.
  - The cost is that GPS and the implicit mixer no longer normalize the same way, so a measured difference between them is not purely the branch structure.

| Code reference | What this code does |
| --- | --- |
| [model.py:450-472](../../../prefill/graph/model.py#L450-L472) | Normalizes each token over its features, scaled and shifted per graph. |
| [model.py:681-725](../../../prefill/graph/model.py#L681-L725) | Places one such normalization after each branch and after the feedforward. |

## D2 — 🟠 Plan gap — the feedforward width is fixed rather than exposed

- **Background:**
  - The plan puts a feedforward block at graph width and lists the settings to expose: architecture, depth, attention heads, and random-feature count.
  - A feedforward block also has an inner width, which the plan does not mention.
- **Decision:** The inner width is twice the graph width and is not a setting.
- **Plan gap or deviation:** The plan neither exposed this width nor fixed it.
- **Reason and tradeoff:**
  - Graph width already controls the block's size, so a second width knob adds experiment surface for little gain.
  - Changing it later means a new setting and a new checkpoint field, which is why the value is named in one place.

| Code reference | What this code does |
| --- | --- |
| [model.py:34-36](../../../prefill/graph/model.py#L34-L36) | Fixes the multiplier in one place, with the reason. |

## D3 — 🟠 Plan gap — GPS drops the hidden-width scale and shift

- **Background:**
  - The implicit mixer normalizes, then applies a learned scale and shift at hidden width, then the activation.
  - The plan says GPS ends with "the same activation as today," without saying whether that scale and shift come with it.
- **Decision:** GPS projects back to hidden width, applies the activation, and scales by the residual weight, with no hidden-width scale and shift.
- **Plan gap or deviation:** The plan's step 7 was ambiguous about which parts of the final stage GPS keeps.
- **Reason and tradeoff:**
  - The last block already applies a learned scale and shift in the graph space, so repeating it at hidden width is redundant.
  - It also saves two parameters per graph per hidden unit, which is the expensive direction here.

| Code reference | What this code does |
| --- | --- |
| [model.py:858-876](../../../prefill/graph/model.py#L858-L876) | Projects to hidden width, applies the activation, scales by the residual weight. |
| [test_gps_mixer.py:618-636](../../../prefill/tests/test_gps_mixer.py#L618-L636) | Fails if the activation or its slope stops being applied. |

## D4 — 🟠 Plan gap — the random-feature draw needed a uniformity correction

- **Background:**
  - Performer's accuracy depends on drawing feature directions uniformly over all directions.
  - Building them from a standard matrix factorization looks correct but is not uniform, because the factorization's sign convention favors some directions.
- **Decision:** The draw folds in the factorization's signs, restoring uniformity.
- **Plan gap or deviation:** The plan specified Performer but not how its features are drawn.
- **Reason and tradeoff:**
  - Without the correction the attention did not converge to softmax attention: its error stalled near 10% however many features were added.
  - With it the error falls as features grow, which is the property this branch is supposed to have.

| Code reference | What this code does |
| --- | --- |
| [model.py:513-538](../../../prefill/graph/model.py#L513-L538) | Draws orthogonal directions and corrects their signs. |
| [test_gps_mixer.py:94-115](../../../prefill/tests/test_gps_mixer.py#L94-L115) | Fails if the estimate stops converging to the exact kernel. |

## D5 — 🟠 Plan gap — the attention branch's numerical conditioning

- **Background:**
  - Performer exponentiates a score before summing, so a large score overflows and a small one underflows.
  - Subtracting a maximum first is safe whenever that maximum is constant across the summed axis, because it then cancels between the attention's numerator and denominator.
- **Decision:**
  - Queries subtract a maximum taken per token; keys subtract one taken over the whole subgraph.
  - The denominator is floored only at the smallest representable value, not at a fixed constant.
- **Plan gap or deviation:** The plan did not cover numerical conditioning of the branch.
- **Reason and tradeoff:**
  - A single subgraph-wide maximum for queries makes a token whose scores sit far below it underflow, which training does in bfloat16.
  - Those same stabilizers leave the denominator on no fixed scale, so an absolute floor clamped healthy values and quietly shrank the output; every feature is a positive exponential, so only underflow needs guarding.

| Code reference | What this code does |
| --- | --- |
| [model.py:600-614](../../../prefill/graph/model.py#L600-L614) | Chooses the axis each maximum is taken over. |
| [model.py:616-639](../../../prefill/graph/model.py#L616-L639) | Floors the denominator at the dtype's smallest value. |
| [test_gps_mixer.py:561-578](../../../prefill/tests/test_gps_mixer.py#L561-L578) | Fails if the stabilizer stops preventing underflow. |

## D6 — 🟠 Plan gap — GPS rejoins streamed token chunks instead of refusing them

- **Background:**
  - Training and validation hand the mixer a span in token-sized pieces, because the implicit mixer can summarize a context piece by piece.
  - A GPS stack cannot consume a span piecewise: its attention and normalization both span the whole subgraph.
- **Decision:** GPS accepts the pieces and rejoins them before running.
- **Plan gap or deviation:** The plan restricted GPS to subgraphs but did not say what happens at this internal streaming entry point, which validation uses every epoch.
- **Reason and tradeoff:**
  - Refusing it would have made GPS training crash at its first validation, since callers reach the mixer this way whether or not they subdivide.
  - The span is a bounded subgraph, so rejoining costs nothing that scoring it did not already cost.

| Code reference | What this code does |
| --- | --- |
| [model.py:905-931](../../../prefill/graph/model.py#L905-L931) | Rejoins the pieces and runs the stack once. |
| [test_gps_mixer.py:282-303](../../../prefill/tests/test_gps_mixer.py#L282-L303) | Checks piecewise delivery gives the same result, and that a short span is rejected. |
| [test_gps_mixer.py:248-279](../../../prefill/tests/test_gps_mixer.py#L248-L279) | Scores a held-out context the way validation does. |

## D7 — 🟠 Plan gap — weight decay is chosen by the architecture

- **Background:**
  - The optimizer applied weight decay to the implicit mixer's two projections and exempted its scale, shift, and residual weight, all named directly.
  - GPS has many more parameters, including three normalizations per block.
- **Decision:** Each architecture reports its own decayed and undecayed groups, and the optimizer uses whatever it reports.
- **Plan gap or deviation:** The plan did not say how the optimizer treats the new parameters.
- **Reason and tradeoff:**
  - Naming GPS parameters in the optimizer would need updating whenever the block changes, and a missed name silently decays a normalization.
  - Both architectures keep the same convention: normalization scales and shifts and the residual weight stay undecayed.

| Code reference | What this code does |
| --- | --- |
| [training.py:325-329](../../../prefill/graph/training.py#L325-L329) | Asks the mixer for its groups instead of naming parameters. |
| [model.py:843-856](../../../prefill/graph/model.py#L843-L856) | Classifies every GPS parameter, so a new one cannot be missed. |
| [test_gps_mixer.py:671-682](../../../prefill/tests/test_gps_mixer.py#L671-L682) | Fails if a normalization starts being decayed. |

## D8 — 🟠 Plan gap — the mixer's correction is returned at a fixed precision

- **Background:**
  - Runs use bfloat16, but the gate's own normalization always returns a higher precision.
  - The implicit mixer returns its correction in that higher precision too, and adding it is what lifts the gate's input to match.
- **Decision:** GPS returns its correction at the same precision the implicit mixer does, and its local branch accumulates its sum over tokens there as well.
- **Plan gap or deviation:**
  - The plan said the gate "receives the same kind of correction it receives now" without stating that precision is part of that contract.
  - Built to the plan as written, GPS returned bfloat16 and every run failed on its first scored subgraph.
- **Reason and tradeoff:**
  - Matching the existing contract is what makes GPS usable at all in the precision runs actually use.
  - It costs a little memory over keeping the whole stack in bfloat16, which is the same trade the implicit mixer already makes.

| Code reference | What this code does |
| --- | --- |
| [model.py:858-876](../../../prefill/graph/model.py#L858-L876) | Returns the correction at the reduction precision. |
| [model.py:668-678](../../../prefill/graph/model.py#L668-L678) | Accumulates the local branch's sum over tokens the same way. |
| [test_gps_mixer.py:685-691](../../../prefill/tests/test_gps_mixer.py#L685-L691) | Fails if the correction is left in the compute precision. |

## D9 — 🟠 Plan gap — a checkpoint records only the architecture settings its run applied

- **Background:**
  - Resuming compares the saved settings against the settings the new run would write, and any difference stops the run.
  - Checkpoints written before this change record no architecture at all, and a gate-only run has no mixer for an architecture to describe.
- **Decision:**
  - A run names its architecture only when it has a mixer, and records the GPS settings only when it uses them.
  - Both scripts fill in the implicit architecture for an older checkpoint that has a mixer but names none.
- **Plan gap or deviation:**
  - The plan promised resume was untouched and that old checkpoints load as implicit, but not how.
  - A first attempt recorded all four settings unconditionally while the fill-in step skipped gate-only checkpoints, so the two sides disagreed and every pre-change gate-only run was stranded.
- **Reason and tradeoff:**
  - Recording only what applies satisfies the same rule the CLI enforces, and makes the two sides agree for old and new checkpoints alike.
  - Naming the architecture whenever there is a mixer also makes warm-starting an old checkpoint as GPS fail immediately, rather than later when implicit weights will not load into a GPS stack.

| Code reference | What this code does |
| --- | --- |
| [train_graph.py:927-935](../../../prefill/train_graph.py#L927-L935) | Records the architecture only with a mixer, and the GPS settings only for GPS. |
| [train_graph_answer.py:352-366](../../../prefill/train_graph_answer.py#L352-L366) | Names the architecture an older checkpoint holds, skipping gate-only ones. |
| [test_gps_mixer.py:1175-1179](../../../prefill/tests/test_gps_mixer.py#L1175-L1179) | Fails if the recorded and filled-in sides stop agreeing. |

## D10 — 🟠 Plan gap — settings that belong to one architecture are refused under the other

- **Background:**
  - A run can train the gate alone, with no mixer, and it already refuses every mixer-only setting rather than ignoring it.
  - Depth, attention heads, and random-feature count apply only to GPS.
- **Decision:** Passing any of them without choosing GPS is an error, and a gate-only run refuses them along with the other mixer settings.
- **Plan gap or deviation:** The plan said depth applies to GPS only but did not say what happens when it is set anyway.
- **Reason and tradeoff:**
  - Accepting them would have written a setting into the checkpoint that nothing applied, which is exactly what the existing gate-only rule forbids.
  - The cost is that switching a command line to the implicit architecture now also means removing those settings.

| Code reference | What this code does |
| --- | --- |
| [train_graph.py:1105-1111](../../../prefill/train_graph.py#L1105-L1111) | Refuses them under another architecture. |
| [train_graph_answer.py:617](../../../prefill/train_graph_answer.py#L617) | Does the same for answer-supervised runs. |
| [train_graph_answer.py:79-91](../../../prefill/train_graph_answer.py#L79-L91) | Adds them to the settings a gate-only run refuses. |
| [test_gps_mixer.py:1222-1233](../../../prefill/tests/test_gps_mixer.py#L1222-L1233) | Fails if they are silently accepted. |

## D11 — 🟠 Plan gap — the recorded activation order names the architecture

- **Background:**
  - A checkpoint records the order its activation is applied in, and loading validates it.
  - GPS applies a different sequence from the implicit mixer.
- **Decision:** Each architecture records its own activation order, and loading checks the one belonging to the architecture the checkpoint claims.
- **Plan gap or deviation:** The plan listed four settings to record and did not mention this existing field, which the new architecture changes the meaning of.
- **Reason and tradeoff:**
  - It gives a second, independent reason a GPS checkpoint cannot be read as implicit, beyond the settings themselves.
  - It is one more field to keep in step if either block's activation sequence ever changes.

| Code reference | What this code does |
| --- | --- |
| [model.py:27](../../../prefill/graph/model.py#L27) | Names the GPS order. |
| [test_gps_mixer.py:425-457](../../../prefill/tests/test_gps_mixer.py#L425-L457) | Round-trips the config a real run writes, so neither field can go missing. |

## D12 — 🔴 Plan deviation — one promised check was replaced, because it cannot hold

- **Background:**
  - The plan's verification list includes "GPS reduces to the current mechanism when its global branch is zeroed."
  - That would hold only if GPS were the implicit mixer plus an attention term.
- **Decision:** That check is not implemented. Its purpose — confirming the branches are wired as described — is covered by checks that each piece of the block is load-bearing and that each graph uses its own weights.
- **Plan gap or deviation:** A verification the plan promised is absent, so it is recorded rather than quietly dropped.
- **Reason and tradeoff:**
  - GPS also normalizes per token (D1), adds a feedforward, and adds a position encoding, so zeroing the attention branch does not recover the implicit mixer; the check would fail on correct code.
  - The replacement checks are narrower per test but, unlike the promised one, they actually fail when a piece of the block is removed.

| Code reference | What this code does |
| --- | --- |
| [test_gps_mixer.py:377-412](../../../prefill/tests/test_gps_mixer.py#L377-L412) | Fails if any per-graph weight stops being selected per graph. |
| [test_gps_mixer.py:474-497](../../../prefill/tests/test_gps_mixer.py#L474-L497) | Fails if evaluation rebuilds a GPS checkpoint as something else. |

## D13 — 🟠 Plan gap — no automatic parameter matching between architectures

- **Background:**
  - The plan notes the two architectures will not have equal parameter counts at the same graph width.
  - It says counts are reported at startup but not whether anything adjusts them.
- **Decision:** Nothing is adjusted automatically; both training scripts print the mixer's parameter count.
- **Plan gap or deviation:** The plan left open whether matching should be automatic.
- **Reason and tradeoff:**
  - Matching counts automatically would silently override the graph width someone asked for.
  - The cost is that a like-for-like comparison needs the widths chosen by hand.

| Code reference | What this code does |
| --- | --- |
| [train_graph.py:1124-1137](../../../prefill/train_graph.py#L1124-L1137) | Prints the count; both scripts call it. |

## D14 — 🟠 Plan gap — the random features are resampled on a schedule set by the run's length

- **Background:**
  - Performer's method resamples its random features periodically during training; ours were drawn once and frozen.
  - With one frozen draw the approximation error is a fixed distortion the model can fit, rather than noise that averages out across training.
- **Decision:**
  - The features are resampled every so many optimizer steps, guarded on training mode, with the counter saved alongside them.
  - The default is the run's planned optimizer steps divided by thirty, not the reference implementations' fixed interval of 1000.
- **Plan gap or deviation:** The plan named Performer but said nothing about resampling; an earlier note here justified freezing them by wanting repeatable evaluation.
- **Reason and tradeoff:**
  - That justification was empty: the references skip the resample outside training, so evaluation is repeatable either way.
  - Runs here are a few hundred optimizer steps against their tens of thousands, so their interval would resample zero times; matching their resample count keeps the intent.
  - Counting optimizer steps rather than forward passes keeps the schedule independent of the memory settings, which change how many forwards one step makes.

| Code reference | What this code does |
| --- | --- |
| [model.py:826-841](../../../prefill/graph/model.py#L826-L841) | Counts steps and resamples on the interval, only while training. |
| [train_graph.py:1140-1155](../../../prefill/train_graph.py#L1140-L1155) | Picks the interval from the run's planned length. |
| [test_gps_mixer.py:694-715](../../../prefill/tests/test_gps_mixer.py#L694-L715) | Covers the schedule, the training-mode guard, and survival across a resume. |

## D15 — 🟠 Plan gap — the random-feature count stays at 32

- **Background:**
  - The feature count trades approximation accuracy against memory and time.
  - Our attention heads are 8 wide, far narrower than the reference's 64, so a given count covers proportionally more of the space.
- **Decision:** Keep 32 as the default.
- **Plan gap or deviation:** The plan exposed the setting but recorded no basis for its default.
- **Reason and tradeoff:**
  - Measured on normalized inputs over 2000 tokens, 32 features at head width 8 recover about 64% of the variation exact softmax attention produces across tokens; the reference's own operating point recovers about 12% at its width.
  - Applying the reference's own sizing rule here would give 16 features, which measures worse, at about 51%.
  - This is a fixed deterministic kernel the model trains around, not softmax attention; the number says how close the two are, not that either is right.

| Code reference | What this code does |
| --- | --- |
| [model.py:33](../../../prefill/graph/model.py#L33) | Sets the default. |

## D16 — 🟠 Plan gap — GPS gets positional information the implicit mixer has none of

- **Background:**
  - GPS adds a sequence position encoding to its input; the implicit mixer has no equivalent and never sees token order.
  - The plan chose the encoding for GPS on its own merits, without noting what it does to the comparison.
- **Decision:** Keep the encoding, and record that the comparison therefore differs in two ways, not one.
- **Plan gap or deviation:** The plan treats the two architectures as differing in structure alone.
- **Reason and tradeoff:**
  - Removing it would make the comparison cleaner but would drop an ingredient the method expects.
  - The cost is that a measured difference between the architectures cannot be attributed to structure alone. Recorded as a known limitation, not fixed here.

| Code reference | What this code does |
| --- | --- |
| [model.py:862-864](../../../prefill/graph/model.py#L862-L864) | Adds the encoding to the stack's input. |

## D17 — 🔴 Plan deviation — the memory rework the review asked for was measured and not done

- **Background:**
  - Review reported that GPS builds its correction at model width across a whole subgraph while the implicit mixer never does, and asked for the correction to be kept at graph width and expanded per token chunk.
  - The implicit mixer's own replay also builds model-width tensors for the same span.
- **Decision:** Measure both paths first, and skip the rework because the premise does not hold.
- **Plan gap or deviation:** A change asked for in review was not made, so it is recorded rather than dropped quietly.
- **Reason and tradeoff:**
  - At a realistic width ratio, one mixer batch leaves the implicit path holding six span-sized model-width tensors and GPS holding five, so GPS is not the heavier of the two.
  - The rework would still reduce GPS's share, but it would make GPS cheaper than the architecture it is being compared against rather than fix a GPS-specific regression, and it is a large change to the training and evaluation loops.
  - Two smaller parts of the same report were real and were done: a full-range token slice no longer copies the correction, and a token budget GPS cannot act on is validated rather than accepted in silence.

| Code reference | What this code does |
| --- | --- |
| [model.py:735-744](../../../prefill/graph/model.py#L735-L744) | Returns the same state instead of copying for a whole-span slice. |
| [model.py:878-903](../../../prefill/graph/model.py#L878-L903) | Validates a token budget it does not otherwise use. |

## D18 — 🔴 Plan deviation — one promised verification is still unmet

- **Background:**
  - The plan promises "Both training loops run end to end on a tiny synthetic model with GPS."
  - The teacher-supervised loop is covered through a full training step.
- **Decision:** The answer-supervised loop is exercised only at the scoring and gradient-replay level, not through its objective and optimizer step.
- **Plan gap or deviation:** A verification the plan promised is partly unmet, and D12 was previously the only deviation recorded, so a reader of this file alone would think exactly one was dropped.
- **Reason and tradeoff:**
  - The pieces GPS changes are the scored subgraphs and their gradients, and those are covered against an independent reference.
  - What stays unverified is the surrounding loop: the objective, the optimizer step, and the logged metrics with a GPS mixer in place. A cluster pilot would exercise it; nothing here does.

| Code reference | What this code does |
| --- | --- |
| [test_gps_mixer.py:347-374](../../../prefill/tests/test_gps_mixer.py#L347-L374) | Covers the answer path's scoring and gradient replay, but not its loop. |
