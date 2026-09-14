"""Build the retention-score appendix, indexed PDF atlas, and curve snapshot.

Uses the same unrounded scores and configuration weighting as the result tables.
Run with the existing plotting environment; --check validates without writing.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean

from render_result_tables import (METHODS, RATIOS, EXTREME_RATIOS, DISPLAY_RATIOS,
                                  RULER, SCBENCH, rows as table_rows)

PAPER = Path(__file__).resolve().parent
SOURCE = PAPER / "data/method-scores.json"
GROUPS = PAPER / "data/retention-plot-groups.json"
DESTINATION = PAPER / "figures/retention-curves"
COLORS = ("#0072B2", "#D55E00", "#009E73")
MARKERS = ("o", "s", "^")
LABELS = ("Graph-based selector", "FastKVzip", "KVzip")


def read(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def balanced_chunks(items, maximum=6):
    count = math.ceil(len(items) / maximum)
    size, extra = divmod(len(items), count)
    start = 0
    for i in range(count):
        stop = start + size + (i < extra)
        yield items[start:stop]
        start = stop


def zoomed_axis(values):
    """Enclose all scores with simple ticks and at least a two-point window."""
    from matplotlib.ticker import MaxNLocator

    low, high = min(values), max(values)
    padding = 0.08 * max(high - low, 2)
    lower, upper = max(0, low - padding), min(100, high + padding)
    if upper - lower < 2:
        lower = max(0, min((lower + upper) / 2 - 1, 98))
        upper = lower + 2
    ticks = MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]).tick_values(lower, upper)
    ticks = [float(round(tick, 8)) for tick in ticks if 0 <= tick <= 100]
    # Extra frame padding keeps markers at 0 or 100 completely visible.
    margin = 0.05 * (ticks[-1] - ticks[0])
    return {"limits": [ticks[0] - margin, ticks[-1] + margin], "ticks": ticks}


def build_pages(source, taxonomy):
    available = set(source["suite"]["tasks"])
    extreme_tasks = set(source["suite"].get("extreme_ratio_tasks", []))
    pages = []

    def ratios_for(tasks):
        """8 points (5% through 100%) only when every task in the group was
        evaluated at the extra ratios; 6 points (20% through 100%) otherwise,
        exactly as before this extension."""
        return DISPLAY_RATIOS if set(tasks) <= extreme_tasks else RATIOS

    def panel(key, title, tasks, subtitle, kind="aggregate"):
        assert tasks and len(tasks) == len(set(tasks)) and set(tasks) <= available, key
        ratio_list = ratios_for(tasks)
        scores = {method: [mean(source["methods"][method]["scores"][task][ratio]
                                for task in tasks) for ratio in ratio_list] for method in METHODS}
        full = {method: curve[-1] for method, curve in scores.items()}
        values = [value for curve in scores.values() for value in curve] + list(full.values())
        return {"id": key, "title": title, "subtitle": subtitle, "kind": kind,
                "tasks": tasks, "ratios": list(ratio_list), "scores": scores, "full_cache": full,
                "shared_full_cache": len(set(full.values())) == 1,
                "y_axis": zoomed_axis(values)}

    def page(section, key, title, panels, note):
        assert 1 <= len(panels) <= 6
        pages.append({"section": section, "id": key, "title": title, "panels": panels,
                      "note": note, "asset": f"{len(pages) + 1:02d}-{key}.pdf"})

    scbase = [f"scbench_{key}" for key in SCBENCH]
    scall = [f"scbench_{key}" + (f"_{variant}" if variant else "")
             for key, (_, variants) in SCBENCH.items() for variant in variants]
    ruler = [f"ruler_{task}_{length}" for length in ("4k", "8k", "64k")
             for task in RULER if f"ruler_{task}_{length}" in available]
    page("Benchmark summaries", "suite-overview", "Benchmark summaries", [
        panel("all-60", "Combined evaluated suite", scall + ruler, "60 configurations; mixed benchmarks"),
        panel("scbench-base", "SCBench: base tasks", scbase, "11 evaluated base tasks; no length variants"),
        panel("scbench-all", "SCBench: all variants", scall, "27 configurations; variants weighted equally"),
        panel("ruler-all", "RULER: evaluated lengths", ruler, "33 configurations across 4K, 8K and 64K"),
    ], "Unweighted configuration means matching the appendix tables. These are summaries of the evaluated suite, not official complete-benchmark results.")
    length_panels = []
    for length in ("4k", "8k", "64k"):
        tasks = [task for task in ruler if task.endswith("_" + length)]
        length_panels.append(panel(f"ruler-{length}", f"RULER {length.upper()}", tasks,
                                   f"{len(tasks)} of 13 tasks evaluated"))
    page("Benchmark summaries", "ruler-lengths", "RULER summaries by context length", length_panels,
         "All 13 tasks are included at 4K and 8K; 64K contains seven selected tasks. Differences across lengths also reflect this unequal task coverage.")

    for variants in (False, True):
        scope = "all-variants" if variants else "base-tasks"
        panels = []
        for group in taxonomy["scbench"]:
            tasks = [f"scbench_{key}" + (f"_{variant}" if variant else "")
                     for key in group["tasks"]
                     for variant in (SCBENCH[key][1] if variants else ("",))]
            subtitle = f"{len(tasks)} configurations" if variants else (
                f"{len(tasks)} of {group['official_task_count']} official base tasks")
            if group["id"] == "semantic-retrieval":
                subtitle += "; Chinese QA absent"
            panels.append(panel(f"scbench-{scope}-{group['id']}", group["label"], tasks, subtitle))
        page("SCBench task families", f"scbench-families-{scope}",
             "SCBench task families: " + ("all variants" if variants else "base tasks"), panels,
             "Categories follow SCBench Section 3.1 and Table 2. Semantic retrieval excludes the unevaluated Chinese-QA task. "
             + ("Prepared length variants inherit their base task's category and receive equal weight."
                if variants else "Only the 11 evaluated base tasks enter these means; prepared length variants are excluded."))

    for family, section, heading in (("ruler", "RULER task families", "task families"),
                                     ("ruler_retrieval_subtypes", "RULER retrieval subtypes", "retrieval subtypes")):
        for length in ("4k", "8k", "64k"):
            panels = []
            for group in taxonomy[family]:
                tasks = [f"ruler_{key}_{length}" for key in group["tasks"]
                         if f"ruler_{key}_{length}" in available]
                if not tasks:
                    continue
                subtitle = f"{len(tasks)} of {group['official_task_count']} configurations evaluated"
                panels.append(panel(f"ruler-{length}-{group['id']}", group["label"], tasks, subtitle))
            note = "Categories follow RULER Section 3. "
            if family == "ruler_retrieval_subtypes":
                note += "Single and multi-key means combine their available variants. "
                if length == "64k":
                    note += "Multi-query retrieval is unevaluated at 64K and has no curve. "
            if length == "64k":
                note += "Only the seven evaluated 64K tasks are included. "
            note += "A one-configuration family reproduces its individual task curve."
            page(section, f"{family.replace('_', '-')}-{length}",
                 f"RULER {length.upper()}: {heading}", panels, note)

    sc_family = {key: group["label"] for group in taxonomy["scbench"] for key in group["tasks"]}
    sc_panels = [panel(task, f"{label} ({variant or 'base'})", [task], sc_family[key], "task")
                 for key, (label, variants) in SCBENCH.items() for variant in variants
                 for task in [f"scbench_{key}" + (f"_{variant}" if variant else "")]]
    for i, chunk in enumerate(balanced_chunks(sc_panels), 1):
        page("SCBench individual tasks", f"scbench-tasks-{i}",
             f"SCBench individual tasks ({i}/5)", chunk,
             "Each panel is one evaluated configuration. Base, mid, short and tiny name the prepared variants used in the tables; mixed tasks retain their deployed combined score.")
    ruler_family = {key: group["label"] for group in taxonomy["ruler"] for key in group["tasks"]}
    for length in ("4k", "8k", "64k"):
        task_panels = []
        for key in RULER:
            task = f"ruler_{key}_{length}"
            if task not in available:
                continue
            label = key.replace("niah_single_", "Single NIAH ").replace("niah_multikey_", "Multi-keys NIAH ")
            label = {"niah_multivalue": "Multi-values NIAH", "niah_multiquery": "Multi-queries NIAH",
                     "vt": "Variable tracking", "cwe": "Common-word extraction", "fwe": "Frequent-word extraction",
                     "qa_1": "QA 1 (SQuAD)", "qa_2": "QA 2 (HotpotQA)"}.get(label, label)
            task_panels.append(panel(task, label, [task], ruler_family[key], "task"))
        chunks = list(balanced_chunks(task_panels))
        for i, chunk in enumerate(chunks, 1):
            page(f"RULER {length.upper()} individual tasks", f"ruler-{length}-tasks-{i}",
                 f"RULER {length.upper()} individual tasks ({i}/{len(chunks)})", chunk,
                 f"Each panel is one evaluated {length.upper()} configuration. " +
                 ("The seven selected 64K tasks are the complete matched subset; the other six have no curves."
                  if length == "64k" else "All 13 RULER task configurations are included across this section."))
    return pages


def check(source, taxonomy, pages):
    assert tuple(source["suite"]["ratios"]) == RATIOS
    tasks = set(source["suite"]["tasks"])
    extreme_tasks = set(source["suite"].get("extreme_ratio_tasks", []))
    assert len(tasks) == 60 and len(METHODS) == 3
    for record in source["methods"].values():
        assert set(record["scores"]) == tasks
        for task, curve in record["scores"].items():
            assert set(RATIOS) <= set(curve) <= set(DISPLAY_RATIOS)
            assert (set(curve) - set(RATIOS)) == (set(EXTREME_RATIOS) if task in extreme_tasks else set())
            assert all(math.isfinite(value) and 0 <= value <= 100 for value in curve.values())
    panels = [panel for page in pages for panel in page["panels"]]
    assert len({panel["id"] for panel in panels}) == len(panels)
    assert Counter(panel["tasks"][0] for panel in panels if panel["kind"] == "task") == Counter(tasks)
    # Panels are extended to 8 points exactly when every one of their tasks
    # is an extreme-ratio task -- nothing more, nothing less.
    extended = [panel for panel in panels if set(panel["ratios"]) == set(DISPLAY_RATIOS)]
    assert {panel["id"] for panel in extended} == {panel["id"] for panel in panels
                                                    if set(panel["tasks"]) <= extreme_tasks}
    summaries = {frozenset(panel["tasks"]): panel for page in pages[:2] for panel in page["panels"]}
    for _, group, aggregate in table_rows(tasks):
        if aggregate:
            panel = summaries[frozenset(group)]
            ratio_list = DISPLAY_RATIOS if set(group) <= extreme_tasks else RATIOS
            assert panel["ratios"] == list(ratio_list)
            for method in METHODS:
                assert panel["scores"][method] == [mean(source["methods"][method]["scores"][task][ratio]
                                                       for task in group) for ratio in ratio_list]
    assert Counter(key for group in taxonomy["scbench"] for key in group["tasks"]) == Counter(SCBENCH.keys())
    assert Counter(key for group in taxonomy["ruler"] for key in group["tasks"]) == Counter(RULER)
    assert Counter(key for group in taxonomy["ruler_retrieval_subtypes"] for key in group["tasks"]) == Counter(
        key for key in RULER if key.startswith("niah_"))
    for panel in panels:
        lower, upper = panel["y_axis"]["limits"]
        values = [value for curve in panel["scores"].values() for value in curve]
        assert lower < min(values) <= max(values) < upper
        assert 2 <= upper - lower <= 110
        for method in METHODS:
            assert panel["full_cache"][method] == panel["scores"][method][-1]
    for values in ([0] * 18, [100] * 18, [99.87, 100], [0, 100]):
        axis = zoomed_axis(values)
        assert axis["limits"][0] < min(values) <= max(values) < axis["limits"][1]
        assert all(0 <= tick <= 100 for tick in axis["ticks"])
    total_points = sum(len(curve) for record in source["methods"].values() for curve in record["scores"].values())
    print(f"PASS: 60 tasks, {total_points:,} raw values ({total_points - 1080} beyond the original "
          f"1,080), seven table aggregates, official partitions; {len(panels)} panels on {len(pages)} "
          f"plot pages ({len(extended)} spanning 5-100%, {len(panels) - len(extended)} spanning "
          "20-100%); zoomed axes include all scores")


def draw_page(page, number, total):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import StrMethodFormatter
    import textwrap

    nrows = math.ceil(len(page["panels"]) / 2)
    fig, axes = plt.subplots(nrows, 2, figsize=(7.2, 1.75 + nrows * 2.05), squeeze=False)
    for ax, panel in zip(axes.flat, page["panels"]):
        x = [100 * float(ratio) for ratio in panel["ratios"]]
        if panel["shared_full_cache"]:
            ax.axhline(next(iter(panel["full_cache"].values())), color="#656B73", linewidth=1,
                       linestyle=(0, (4, 3)), zorder=1)
        else:
            for i, (method, color) in enumerate(zip(METHODS, COLORS)):
                ax.axhline(panel["full_cache"][method], color=color, linewidth=1,
                           linestyle=(i * 3, (3, 6)), zorder=1)
        for i, (method, color, marker) in enumerate(zip(METHODS, COLORS, MARKERS)):
            ax.plot(x, panel["scores"][method], color=color, marker=marker,
                    linewidth=1.5, markersize=4.1 - 0.4 * i, zorder=3 + i)
        ax.set_title(panel["title"], fontsize=10, pad=20, loc="left", fontweight="normal")
        ax.text(0, 1.04, panel["subtitle"], transform=ax.transAxes, fontsize=7.2, color="#505760")
        ax.set_xlim(x[0] - 3, x[-1] + 3)
        ax.set_ylim(panel["y_axis"]["limits"])
        ax.set_xticks(x, [str(round(value)) for value in x])
        ax.set_yticks(panel["y_axis"]["ticks"])
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        ax.tick_params(labelsize=8, length=3)
        ax.grid(axis="y", color="#E2E5E9", linewidth=0.6)
        values = [value for curve in panel["scores"].values() for value in curve]
        if min(values) == max(values):
            ax.text(0.5, 0.45, f"All methods: {values[0]:g}", transform=ax.transAxes,
                    ha="center", fontsize=8, color="#505760")
    for ax in list(axes.flat)[len(page["panels"]):]:
        ax.set_visible(False)
    legend = [Line2D([], [], color=color, marker=marker, linewidth=1.5, markersize=4, label=label)
              for color, marker, label in zip(COLORS, MARKERS, LABELS)]
    fig.suptitle(page["title"], x=0.08, y=0.986, ha="left", fontsize=12, fontweight="bold")
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.54, 0.94), ncol=3,
               frameon=False, fontsize=9, handlelength=2.5, columnspacing=1.6)
    fig.text(0.54, 0.871, "Dashed: full cache (method color; gray if shared). Y-axis range varies by panel.",
             ha="center", fontsize=7.7, color="#505760")
    fig.supylabel("Score (zoomed)", x=0.009, fontsize=10)
    fig.supxlabel("Retention ratio (%)", y=0.095, fontsize=10)
    fig.text(0.08, 0.028, textwrap.fill(page["note"], 115), fontsize=7.5, va="bottom", linespacing=1.4)
    fig.text(0.986, 0.01, f"{number:02d} / {total:02d}", fontsize=7, ha="right", color="#69727E")
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.19, top=0.79, hspace=0.62, wspace=0.24)
    return fig


def tex_escape(value):
    return value.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def write_appendix(pages):
    panels = [panel for page in pages for panel in page["panels"]]
    extended = [panel for panel in panels if set(panel["ratios"]) == set(DISPLAY_RATIOS)]
    extreme_note = (f"{len(extended)} of {len(panels)} panels additionally include 5 and 10\\% "
                     "retention: the 11 SCBench base tasks, their aggregate mean, and their four "
                     "family panels. These two extra points were evaluated only for the 11 SCBench "
                     "base tasks; the 16 length variants and all RULER configurations are unaffected "
                     "and still span 20--100\\%.")
    lines = [r"% Generated by plot_retention_scores.py; edit the generator instead.",
             r"\section{Retention--Score Curves}", r"\label{app:retention-curves}", "",
             "This appendix plots all 60 matched configurations and their benchmark and task-family summaries. "
             "The seven benchmark summaries reproduce the aggregate rows in Tables~\\ref{tab:results-graphkv}--\\ref{tab:results-kvzip}. "
             "Each aggregate is the arithmetic mean of its unrounded configuration scores, independently at each retention ratio. "
             "All plots use absolute native-evaluator scores (0--100) and the same method colors. "
             "Each panel has a zoomed linear y-axis enclosing all three methods and their full-cache baselines. "
             "Y-axis ranges vary across panels; tick labels show the actual scores. "
             "Flat and near-flat panels span at least two score points, and identical constant curves are explicitly marked.", "",
             "Solid curves show the three methods at 20, 30, 40, 50, 75, and 100\\% retention. " + extreme_note + " "
             "Dashed horizontal lines show each method's own measured full-cache score, using the matching color; "
             "one gray dashed line is used when all three full-cache scores coincide. Coincident curves and nearly equal baselines may overlap. "
             "The full-cache point is included at 100\\% retention. "
             "SCBench base-task and all-variant means are distinguished; RULER 64K contains only seven selected tasks.", "",
             "SCBench families follow Section~3.1 and Table~2 of \\citet{li2025scbench}: "
             "string retrieval, semantic retrieval, global information processing, and multi-tasking. "
             "Chinese QA is absent from our semantic-retrieval subset. Prepared length variants inherit their base task's family. "
             "RULER families and needle-retrieval subtypes follow Section~3 of \\citet{hsieh2024ruler}. "
             "One-task families repeat their individual curve for convenient comparison. "
             "The deployed combined scores for mixed SCBench and Many-shot tasks are retained; component scores are not separately available in the table snapshot. "
             "Unevaluated configurations have no curves.", "",
             r"\paragraph{Plot guide.}"]
    sections = dict.fromkeys(page["section"] for page in pages)
    guides = []
    for i, section in enumerate(sections, 1):
        selected = [page for page in pages if page["section"] == section]
        guides.append(tex_escape(section) + r" (Figures~\ref{fig:retention-" + selected[0]["id"] +
                      r"}--\ref{fig:retention-" + selected[-1]["id"] + "})")
    lines += ["; ".join(guides) + ".", r"\clearpage", ""]
    previous = None
    for page in pages:
        if page["section"] != previous:
            lines += [r"\subsection{" + tex_escape(page["section"]) + "}", ""]
            previous = page["section"]
        lines += [r"\begin{figure}[!ht]", r"\centering",
                  r"\includegraphics[width=\linewidth]{figures/retention-curves/" + page["asset"] + "}",
                  r"\caption{" + tex_escape(page["title"]) + ".}",
                  r"\label{fig:retention-" + page["id"] + "}", r"\end{figure}", r"\clearpage", ""]
    (PAPER / "sections/appendix_retention_curves.tex").write_text("\n".join(lines))


def write_atlas(pages):
    from pypdf import PdfWriter
    writer = PdfWriter()
    previous, parent = None, None
    for i, page in enumerate(pages):
        writer.append(DESTINATION / page["asset"], import_outline=False)
        if page["section"] != previous:
            parent = writer.add_outline_item(page["section"], i)
            previous = page["section"]
        writer.add_outline_item(page["title"], i, parent=parent)
    writer.add_metadata({"/Title": "Retention-score plots: benchmark summaries and all evaluated tasks",
                         "/Subject": "Absolute scores; three methods and measured full-cache references"})
    writer.write(PAPER.parents[1] / "pdf/retention-score-atlas.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source, taxonomy = read(SOURCE), read(GROUPS)
    pages = build_pages(source, taxonomy)
    check(source, taxonomy, pages)
    if args.check:
        return
    os.environ.setdefault("MPLCONFIGDIR", str(PAPER / "build/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for i, page in enumerate(pages, 1):
        fig = draw_page(page, i, len(pages))
        fig.savefig(DESTINATION / page["asset"], metadata={"Title": page["title"]})
        plt.close(fig)
    extreme_tasks = sorted(set(source["suite"].get("extreme_ratio_tasks", [])))
    snapshot = {"source": str(SOURCE.relative_to(PAPER)), "source_sha256": sha256(SOURCE),
                "taxonomy_sha256": sha256(GROUPS), "ratios": list(RATIOS),
                "extreme_ratios": list(EXTREME_RATIOS), "extreme_ratio_tasks": extreme_tasks,
                "weighting": "Unweighted arithmetic mean of full-precision configuration scores, separately at each ratio.",
                "baseline": "Each method's own full-cache score; a shared gray line only when values are equal.",
                "y_axis": "Panel-specific linear limits include all method scores and full-cache baselines. "
                          "Use 8% data padding, a minimum two-point window, simple absolute-score ticks, "
                          "and 5% frame padding beyond the extreme ticks.",
                "extreme_ratio_note": "A panel spans 5-100% only when every task it plots is one of "
                                      "extreme_ratio_tasks; otherwise it spans 20-100%, unchanged.",
                "pages": pages}
    (PAPER / "data/retention-plot-scores.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    write_appendix(pages)
    write_atlas(pages)
    n_extended = sum(1 for page in pages for panel in page["panels"]
                     if set(panel["ratios"]) == set(DISPLAY_RATIOS))
    n_panels = sum(len(page["panels"]) for page in pages)
    index = ["# Retention-score plot index", "",
             "All scores and task memberships are archived in [the plot snapshot](data/retention-plot-scores.json). "
             "The standalone atlas has PDF bookmarks for each section and page. Each panel uses raw scores, "
             "three solid method curves and measured full-cache dashed references. Each panel has a zoomed linear y-axis; "
             "ranges vary by panel, and ticks show actual scores. All scores and baselines remain visible. "
             "Identical constant curves are labeled. The tables and relative context-length Figure 2 are unchanged.", "",
             f"{n_extended} of {n_panels} panels (marked † below) additionally plot 5% and 10% "
             "retention: the 11 SCBench base tasks, their aggregate mean, and their four family panels. "
             "Every other panel is unaffected and still spans 20-100%.", "",
             "Official grouping sources: [SCBench](https://arxiv.org/html/2412.10319v2#S3) and "
             "[RULER](https://arxiv.org/html/2404.06654v3#S3). Category mappings and coverage notes are in "
             "[the grouping snapshot](data/retention-plot-groups.json).", ""]
    previous = None
    for i, page in enumerate(pages, 1):
        if page["section"] != previous:
            index += ["## " + page["section"], ""]
            previous = page["section"]
        index += [f"### Atlas page {i}: [{page['title']}](figures/retention-curves/{page['asset']})", "",
                  "| Panel | Configuration(s) |", "|---|---|"]
        index += ["| " + panel["title"] + (" †" if set(panel["ratios"]) == set(DISPLAY_RATIOS) else "")
                  + " | " + ", ".join(f"`{task}`" for task in panel["tasks"]) + " |"
                  for panel in page["panels"]]
        index += ["", page["note"], ""]
    (PAPER / "RETENTION_PLOTS.md").write_text("\n".join(index))


if __name__ == "__main__":
    main()
