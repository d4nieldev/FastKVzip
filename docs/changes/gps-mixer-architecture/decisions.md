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
| [model.py:1136-1158](../../../prefill/graph/model.py#L1136-L1158) | Normalizes each token over its features, scaled and shifted per graph. |
| [model.py:1367-1411](../../../prefill/graph/model.py#L1367-L1411) | Places one such normalization after each branch and after the feedforward. |

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
| [model.py:41-43](../../../prefill/graph/model.py#L41-L43) | Fixes the multiplier in one place, with the reason. |

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
| [model.py:1551-1569](../../../prefill/graph/model.py#L1551-L1569) | Projects to hidden width, applies the activation, scales by the residual weight. |
| [test_gps_mixer.py:629-647](../../../prefill/tests/test_gps_mixer.py#L629-L647) | Fails if the activation or its slope stops being applied. |

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
| [model.py:1199-1224](../../../prefill/graph/model.py#L1199-L1224) | Draws orthogonal directions and corrects their signs. |
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
| [model.py:1286-1300](../../../prefill/graph/model.py#L1286-L1300) | Chooses the axis each maximum is taken over. |
| [model.py:1302-1325](../../../prefill/graph/model.py#L1302-L1325) | Floors the denominator at the dtype's smallest value. |
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
| [model.py:1604-1640](../../../prefill/graph/model.py#L1604-L1640) | Rejoins the pieces and runs the stack once. |
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
| [training.py:336-340](../../../prefill/graph/training.py#L336-L340) | Asks the mixer for its groups instead of naming parameters. |
| [model.py:1536-1549](../../../prefill/graph/model.py#L1536-L1549) | Classifies every GPS parameter, so a new one cannot be missed. |
| [test_gps_mixer.py:682-693](../../../prefill/tests/test_gps_mixer.py#L682-L693) | Fails if a normalization starts being decayed. |

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
| [model.py:1551-1569](../../../prefill/graph/model.py#L1551-L1569) | Returns the correction at the reduction precision. |
| [model.py:1354-1364](../../../prefill/graph/model.py#L1354-L1364) | Accumulates the local branch's sum over tokens the same way. |
| [test_gps_mixer.py:696-702](../../../prefill/tests/test_gps_mixer.py#L696-L702) | Fails if the correction is left in the compute precision. |

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
| [train_graph.py:997-1005](../../../prefill/train_graph.py#L997-L1005) | Records the architecture only with a mixer, and the GPS settings only for GPS. |
| [model.py:126-142](../../../prefill/graph/model.py#L126-L142) | Names the architecture an older checkpoint holds, skipping gate-only ones. |
| [test_gps_mixer.py:1288-1292](../../../prefill/tests/test_gps_mixer.py#L1288-L1292) | Fails if the recorded and filled-in sides stop agreeing. |

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
| [train_graph.py:1188-1194](../../../prefill/train_graph.py#L1188-L1194) | Refuses them under another architecture. |
| [train_graph_answer.py:626](../../../prefill/train_graph_answer.py#L626) | Does the same for answer-supervised runs. |
| [train_graph_answer.py:80-92](../../../prefill/train_graph_answer.py#L80-L92) | Adds them to the settings a gate-only run refuses. |
| [test_gps_mixer.py:1335-1346](../../../prefill/tests/test_gps_mixer.py#L1335-L1346) | Fails if they are silently accepted. |

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
| [model.py:30](../../../prefill/graph/model.py#L30) | Names the GPS order. |
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
| [train_graph.py:1245-1258](../../../prefill/train_graph.py#L1245-L1258) | Prints the count; both scripts call it. |

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
| [model.py:1514-1529](../../../prefill/graph/model.py#L1514-L1529) | Counts steps and resamples on the interval, only while training. |
| [train_graph.py:1261-1276](../../../prefill/train_graph.py#L1261-L1276) | Picks the interval from the run's planned length. |
| [test_gps_mixer.py:705-726](../../../prefill/tests/test_gps_mixer.py#L705-L726) | Covers the schedule, the training-mode guard, and survival across a resume. |

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
| [model.py:40](../../../prefill/graph/model.py#L40) | Sets the default. |

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
| [model.py:1555-1557](../../../prefill/graph/model.py#L1555-L1557) | Adds the encoding to the stack's input. |

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
| [model.py:1421-1430](../../../prefill/graph/model.py#L1421-L1430) | Returns the same state instead of copying for a whole-span slice. |
| [model.py:1571-1602](../../../prefill/graph/model.py#L1571-L1602) | Validates a token budget it does not otherwise use. |

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

---

*The entries below cover rebasing this branch onto the merged granola
normalization work. The approved plan for that rebase is appended to
[plan.md](plan.md).*

## D19 — 🟠 Plan gap — GPS refuses a normalization mode and records none

- **Background:**
  - The merged granola work makes the implicit mixer's normalization a choice of none, batchnorm, or granola.
  - A GPS stack normalizes inside each of its own blocks, so none of those modes runs when it is selected.
- **Decision:** Asking for batchnorm or granola together with GPS is an error, and a GPS run records `none`.
- **Plan gap or deviation:** The original plan predates the normalization setting existing.
- **Reason and tradeoff:**
  - Accepting the setting would record one nothing applied, which is the rule both scripts already enforce for the GPS settings.
  - The cost is that GPS cannot be compared against the implicit mixer under granola normalization until someone makes GPS honour it; an architecture comparison runs at one normalization setting for the implicit side.

| Code reference | What this code does |
| --- | --- |
| [train_graph.py:1209-1232](../../../prefill/train_graph.py#L1209-L1232) | Refuses a mode GPS cannot apply and settles on none. |
| [test_gps_mixer.py:1068-1096](../../../prefill/tests/test_gps_mixer.py#L1068-L1096) | Fails if either script accepts one. |

## D20 — 🟠 Plan gap — each mixer reports the interface the shared code asks for

- **Background:**
  - GraNoLa draws random features per call from a seed, and the trainer settles one seed per context by asking the scorer, which asks the mixer which normalization it uses.
  - Loading a checkpoint compared seven normalization settings against attributes on the mixer.
- **Decision:**
  - A GPS mixer reports its normalization as `none`, and accepts the seed arguments callers forward rather than failing on them.
  - Each architecture reports the normalization settings it actually applies, and the loader compares those.
- **Plan gap or deviation:**
  - The plan predates all of this machinery.
  - Without it a GPS run raised inside the seed guard before any check ran, and again while loading, so it could not train or evaluate at all.
- **Reason and tradeoff:**
  - Reporting through the mixer keeps the shared code free of architecture branches, as the weight-decay grouping already is.
  - Comparing only what a mixer applies means a GPS checkpoint's inert GraNoLa fields are not checked; nothing reads them.

| Code reference | What this code does |
| --- | --- |
| [model.py:1097-1108](../../../prefill/graph/model.py#L1097-L1108) | The implicit mixer reports all seven settings. |
| [training.py:523-525](../../../prefill/graph/training.py#L523-L525) | The loader compares what the mixer reports. |
| [test_gps_mixer.py:1122-1130](../../../prefill/tests/test_gps_mixer.py#L1122-L1130) | Fails if the seed guard stops answering for GPS. |

## D21 — 🟠 Plan gap — one canonicaliser, composed rather than extended

- **Background:**
  - Granola added a helper that fills in the settings an older checkpoint lacks, shared by both training scripts and the evaluator.
  - It returns early for a checkpoint that already names a normalization.
- **Decision:** A composed entry point fills the architecture first, then calls granola's helper. This branch's own back-fills are deleted.
- **Plan gap or deviation:** This branch had solved the same problem separately, in three places.
- **Reason and tradeoff:**
  - Adding the architecture inside granola's helper would have missed every granola-era checkpoint, since those already name a normalization and take its early return.
  - One entry point means all three consumers agree about what an older checkpoint means, which is the property that keeps resume working.

| Code reference | What this code does |
| --- | --- |
| [model.py:126-142](../../../prefill/graph/model.py#L126-L142) | Fills the architecture, then delegates. |
| [test_gps_mixer.py:1133-1157](../../../prefill/tests/test_gps_mixer.py#L1133-L1157) | Covers a granola-era checkpoint, which neither branch had seen. |

## D22 — ⚪ Superseded — the contiguous-slice helper is gone

- **Background:** This branch added a way to slice a prepared state without copying, and routed two hand-built call sites through the prepared object.
- **Decision:** Deleted. Granola independently did the same work, and its version also handles the fields it added.
- **Plan gap or deviation:** Supersedes the second half of D17. Its remaining part, the whole-span slice returning itself rather than copying, still stands for the GPS state.
- **Reason and tradeoff:** Keeping two ways to slice the same object would leave the one that does not understand granola's state as a trap.

---

*The entries below come from the review of the merge. The review's other items
were fixes with no choice in them, so they carry no entry.*

## D23 — 🟠 Plan gap — an older checkpoint belongs to the implicit mixer only

- **Background:**
  - Loading a checkpoint into a running scorer compares the activation marker the checkpoint recorded against the one the running code expects.
  - The merge made the expected marker depend on the architecture, and read that architecture out of the checkpoint being checked, so the check compared the checkpoint against itself.
  - Checkpoints written before the normalization setting existed record an older marker, which that check has always tolerated.
- **Decision:**
  - The expected marker comes from the architecture the scorer runs.
  - The older marker is tolerated only for an implicit scorer, so a GPS scorer refuses those checkpoints.
- **Plan gap or deviation:** The plan said to settle the activation field across the two architectures but not what it is compared against, and the tolerance predates GPS existing.
- **Reason and tradeoff:**
  - Without it, a checkpoint built for the other architecture is refused only when its weights are looked up by name, as a raw error listing tensors, from a function whose callers only catch the other kind.
  - Every pre-normalization checkpoint is an implicit mixer, so a GPS scorer has none of them to accept; tolerating their marker would only let one through to that same raw failure.

| Code reference | What this code does |
| --- | --- |
| [training.py:452-495](../../../prefill/graph/training.py#L452-L495) | Expects the scorer's marker, and narrows the older one to implicit. |
| [test_gps_mixer.py:1410-1450](../../../prefill/tests/test_gps_mixer.py#L1410-L1450) | Fails if either architecture's checkpoint reaches the other's scorer. |
| [test_gps_mixer.py:1453-1468](../../../prefill/tests/test_gps_mixer.py#L1453-L1468) | Fails if a pre-normalization checkpoint is accepted into a GPS scorer. |

## D24 — 🟠 Plan gap — GPS refuses the GraNoLa settings, but its checkpoint still holds their defaults

- **Background:**
  - D19 made asking for a normalization mode with GPS an error.
  - The five remaining settings of that shape — the sharing mode and the four GraNoLa ones — were still accepted, and written into a GPS checkpoint that nothing reads.
- **Decision:**
  - Passing any of the five together with GPS is an error, in both training scripts.
  - Their default values are still written into a GPS checkpoint. They are not dropped.
- **Plan gap or deviation:** D19 covered one setting of the group and left the other five unstated.
- **Reason and tradeoff:**
  - Refusing them is what the README already claimed, and it means nobody chooses a setting believing it ran.
  - Dropping the values as well would be the stricter rule, but the shared canonicaliser requires all seven once a checkpoint names a normalization, so a file without them would stop loading. That is a change to code both features share, and it is not worth making for values no reader consults.

| Code reference | What this code does |
| --- | --- |
| [train_graph.py:1176-1203](../../../prefill/train_graph.py#L1176-L1203) | Names the five, and refuses them under GPS. |
| [test_gps_mixer.py:1580-1614](../../../prefill/tests/test_gps_mixer.py#L1580-L1614) | Fails if either script accepts one, or if the implicit mixer stops taking them. |

## D25 — 🟠 Plan gap — the redraw interval is validated but stays optional

- **Background:**
  - A GPS checkpoint must record its depth, its attention heads and its feature count; each is checked while validating, because each decides a tensor shape.
  - The redraw interval was checked nowhere, and was first read deep inside rebuilding, after the language model had already been loaded.
- **Decision:** It is validated with the other three, but a checkpoint without it still loads and means never redraw.
- **Plan gap or deviation:** The plan did not say which GPS settings a checkpoint must carry.
- **Reason and tradeoff:**
  - It decides no tensor shape, and zero is already the value that means never, so demanding it would refuse files that are complete in every way that matters.
  - The cost is one more setting whose absence and whose zero cannot be told apart. Nothing distinguishes them.

| Code reference | What this code does |
| --- | --- |
| [evaluation.py:278-289](../../../prefill/graph/evaluation.py#L278-L289) | Checks the value where the other three are checked, and lets it be absent. |
| [test_gps_mixer.py:1559-1577](../../../prefill/tests/test_gps_mixer.py#L1559-L1577) | Fails if a bad value survives validation, or a missing one stops loading. |
