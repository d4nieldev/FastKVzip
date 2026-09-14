# Extreme retention ratios (5%, 10%) for SCBench base tasks

User request (verbatim, across two turns):

> I want to add the full results to the paper, update the tables and figures

> we should extend the existing tables with more columns for the new ratios
> and update the relevant figures with the retention ratio / score curves
> adding 6 more points (2 ratios for each method) for each of the scbench
> figures

Context: 33 new cluster evaluations (3 methods x 11 SCBench base tasks x 2
ratios) had just finished and uploaded to the same three W&B runs the paper
already cites. The first turn was ambiguous about mechanism; the user's
second message settled it explicitly: extend the existing tables/figures in
place rather than add a separate supplementary table.

Implementation scope (agent plan, approved before implementation): the
paper's pipeline (`render_result_tables.py`, `plot_retention_scores.py`) is
built around a fixed 60-configuration x 6-ratio grid with hard coverage
assertions. The new data covers only 11 of those 60 configurations. Extend
in place means: those 11 tasks, and any row/panel whose group is composed
*entirely* of those 11, get two new points; everything else keeps exactly
what it has today, shown as an explicit "—" in the new table columns rather
than a fabricated or partial mean (plots don't need a dash concept: a group
either qualifies for both new points or keeps its original 6, since partial
per-task point sets aren't meaningful on a curve). Figure 2 (relative
context-length pooling) is out of scope: it needs raw per-context scores,
which are not available for the two new ratios, only W&B's aggregated task
means.
