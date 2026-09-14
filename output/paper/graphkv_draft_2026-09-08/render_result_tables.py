"""Render the three appendix/Markdown tables from the verified score snapshot."""

import json
import math
from pathlib import Path
from statistics import mean


PAPER = Path(__file__).resolve().parent
RATIOS = ("0.2", "0.3", "0.4", "0.5", "0.75", "1.0")
# 5% and 10% retention were additionally evaluated for the 11 SCBench base
# tasks only (no length variants, no RULER). DISPLAY_RATIOS is the full
# column order; RATIOS remains the floor every one of the 60 configurations
# must have.
EXTREME_RATIOS = ("0.05", "0.1")
DISPLAY_RATIOS = EXTREME_RATIOS + RATIOS
SCBENCH = {
    "kv": ("KV", ("", "mid", "short", "tiny")),
    "mf": ("MF", ("", "mid", "short", "tiny")),
    "qa_eng": ("QA (English)", ("",)),
    "repoqa": ("RepoQA", ("", "short", "tiny")),
    "summary": ("Summary", ("", "mid", "short", "tiny")),
    "choice_eng": ("Choice (English)", ("",)),
    "many_shot": ("Many-shot", ("", "short", "tiny")),
    "prefix_suffix": ("Prefix/suffix", ("", "mid", "short", "tiny")),
    "vt": ("VT", ("",)),
    "summary_with_needles": ("Summary + needles", ("",)),
    "repoqa_and_kv": ("RepoQA + KV", ("",)),
}
RULER = ("niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1",
         "niah_multikey_2", "niah_multikey_3", "niah_multivalue", "niah_multiquery",
         "vt", "cwe", "fwe", "qa_1", "qa_2")
METHODS = {
    "graphkv": ("GraphKV (ours)", "Our graph-based selector",
                "post-prefill pruning; 2% protected local window"),
    "fastkvzip": ("Official FastKVzip", "Fast KVzip",
                  "native chunked-prefill eviction; 2% protected local window"),
    "kvzip": ("KVzip", "KVzip",
              "native post-prefill reconstruction scoring; no protected local window"),
}


def rows(tasks):
    selected = []
    for key, (label, variants) in SCBENCH.items():
        for variant in variants:
            task = "scbench_" + key + ("_" + variant if variant else "")
            selected.append(task)
            yield "SCBench " + label + (f" ({variant})" if variant else ""), [task], False
    assert len(selected) == 27 and set(selected) <= set(tasks)
    yield "SCBench 11-base-task mean", ["scbench_" + key for key in SCBENCH], True
    yield "SCBench 27-configuration mean", selected, True
    yield None, [], False
    ruler = []
    for length in ("4k", "8k", "64k"):
        group = []
        for key in RULER:
            task = f"ruler_{key}_{length}"
            if task not in tasks:
                continue
            group.append(task)
            label = key.replace("_", " ")
            for acronym in ("niah", "vt", "cwe", "fwe", "qa"):
                label = label.replace(acronym, acronym.upper())
            yield f"RULER {length.upper()} {label}", [task], False
        ruler += group
        suffix = "selected 7 mean" if length == "64k" else "13-task mean"
        yield f"RULER {length.upper()} {suffix}", group, True
    assert len(ruler) == 33 and set(selected + ruler) == set(tasks)
    yield "RULER selected 33 mean", ruler, True
    yield "Overall selected 60 mean", selected + ruler, True


def render(source, figure):
    tasks = source["suite"]["tasks"]
    assert len(tasks) == 60 and tuple(source["suite"]["ratios"]) == RATIOS
    extreme_tasks = set(source["suite"].get("extreme_ratio_tasks", []))
    assert extreme_tasks <= set(tasks)
    inventory = json.loads((PAPER / "data/benchmark-context-lengths.json").read_text())["tasks"]
    contexts = sum(inventory[task]["contexts"] for task in tasks)
    questions = sum(inventory[task]["questions"] for task in tasks)
    assert (contexts, questions) == (18531, 28236)
    scope = ("Bold aggregate rows are arithmetic means of unrounded configuration scores, "
             "computed independently for each column. SCBench 11-base-task mean excludes "
             "the 16 length variants; SCBench 27-configuration mean includes them equally. "
             "RULER selected 33 weights the 4K, 8K, and selected 64K groups by 13/33, "
             "13/33, and 7/33. These are descriptive prepared-split summaries, not official "
             "complete-benchmark scores; the overall mean also mixes heterogeneous metrics. "
             "Display values are rounded only after aggregation.")
    extreme_note = ("The 5% and 10% columns are additionally reported for the 11 SCBench "
                     "base-task configurations and their aggregate mean only. The 16 SCBench "
                     "length variants and all 33 RULER configurations were not evaluated at "
                     "these two ratios and show — in both columns, as do the aggregate "
                     "rows that include them.")
    # "%" is a LaTeX comment character -- unescaped, it silently swallows the
    # rest of the line. Escape it here in addition to the em-dash fix above.
    extreme_note_latex = extreme_note.replace("—", "---").replace("%", r"\%")
    markdown = ["# Complete method comparison", "",
                "All scores are absolute scores on a 0-100 scale; higher is better. Columns "
                "are retained-context ratios, not averages over ratios. The full-cache column "
                "is measured independently for each method. The same suite has 60 configurations, "
                f"{contexts:,} contexts, and {questions:,} questions per method.", "", scope, "",
                extreme_note, "",
                "All methods use Qwen2.5-7B-Instruct-1M. FastKVzip uses its official gate and "
                "native 16,000-token chunked eviction; GraphKV and KVzip prune after full-context "
                "prefill. GraphKV and FastKVzip have a 0.02 protected window; KVzip has none. "
                "The inherited 48-token generation limit for Summary + needles applies to both "
                "summary and retrieval questions. Mixed-task configurations retain one combined "
                "score each. This compares deployed method pipelines, not just architectures.", ""]
    latex = [r"\section{Detailed Benchmark Results}", r"\label{app:benchmark-results}", "",
             "Each table contains all 60 evaluated configurations. " + scope,
             extreme_note_latex, "Full-cache scores are method-specific.", ""]
    col_labels = ["5\\%", "10\\%", "20\\%", "30\\%", "40\\%", "50\\%", "75\\%", "Full (100\\%)"]
    header = r"Benchmark / configuration & " + " & ".join(col_labels) + r" \\"
    col_labels_md = ["5%", "10%", "20%", "30%", "40%", "50%", "75%", "Full (100%)"]
    n_cols = 1 + len(DISPLAY_RATIOS)
    for method, (title, caption, protocol) in METHODS.items():
        record = source["methods"][method]
        scores = record["scores"]
        assert set(scores) == set(tasks)
        for task, curve in scores.items():
            assert set(RATIOS) <= set(curve) <= set(DISPLAY_RATIOS)
            assert all(math.isfinite(v) and 0 <= v <= 100 for v in curve.values())
            assert (set(curve) - set(RATIOS)) == (set(EXTREME_RATIOS) if task in extreme_tasks else set())
        markdown += [f"## {title}", "", f"{protocol}. Source: [{record['run_id']}]({record['url']}).", "",
                     "| Benchmark / configuration | " + " | ".join(col_labels_md) + " |",
                     "|---|" + "---:|" * len(DISPLAY_RATIOS)]
        latex += [r"\begingroup", r"\small", r"\setlength{\tabcolsep}{4pt}",
                  r"\begin{longtable}{@{}l" + "r" * len(DISPLAY_RATIOS) + r"@{}}",
                  "\\caption{" + caption + ": " + protocol.replace("%", r"\%") +
                  r". Absolute scores (0--100) for the complete selected suite.}",
                  "\\label{tab:results-" + method + r"} \\", r"\hline", header,
                  r"\hline", r"\endfirsthead",
                  "\\multicolumn{" + str(n_cols) + r"}{l}{\tablename\ \thetable{} (continued)} \\",
                  r"\hline", header, r"\hline", r"\endhead", r"\hline",
                  "\\multicolumn{" + str(n_cols) + r"}{r}{\emph{Continued on next page}} \\", r"\endfoot",
                  r"\hline", r"\endlastfoot"]
        for label, group, aggregate in rows(tasks):
            if label is None:
                latex += [r"\pagebreak[4]"]
                continue
            cells = [label]
            for r in DISPLAY_RATIOS:
                if all(r in scores[t] for t in group):
                    cells.append(f"{mean(scores[t][r] for t in group):.2f}")
                else:
                    cells.append("—")
            markdown += ["| " + " | ".join(f"**{v}**" if aggregate else v for v in cells) + " |"]
            # The document declares no inputenc/fontenc; use the classic TeX
            # em-dash ligature rather than a literal Unicode character.
            latex_cells = [v.replace("—", "---") for v in cells]
            if aggregate:
                latex += [r"\hline"]
            latex += [" & ".join(r"\textbf{" + v + "}" if aggregate else v for v in latex_cells) + r" \\"]
            if aggregate:
                latex += [r"\hline"]
        markdown += [""]
        latex += [r"\end{longtable}", r"\endgroup", r"\clearpage", ""]
    total_points = sum(len(curve) for method in METHODS for curve in source["methods"][method]["scores"].values())
    extension = source.get("extreme_ratio_extension")
    extension_note = (f" 5%/10% values for the 11 SCBench base tasks were added "
                       f"{extension['added_at']}, sourced from the same three runs." if extension else "")
    markdown += [f"Verified source snapshot: {source['checked_at']}. All 180 method/configuration "
                 "curves are complete at the original six ratios (1,080 raw score values), with no "
                 "conflicting values, plus 66 additional 5%/10% values for the 11 SCBench base tasks "
                 f"across 3 methods ({total_points:,} raw score values in total).{extension_note} "
                 "See [the full-precision source snapshot](data/method-scores.json) for source "
                 "identities and pinned dataset revisions."]
    latex.append(figure)
    return "\n".join(latex) + "\n", "\n".join(markdown) + "\n"


if __name__ == "__main__":
    import difflib
    source = json.loads((PAPER / "data/method-scores.json").read_text())
    target = PAPER / "sections/appendix_results.tex"
    figure = target.read_text().split(r"\begin{figure}", 1)[1]
    figure = r"\begin{figure}" + figure.strip()
    latex, markdown = render(source, figure)
    # Emit an apply_patch patch, keeping edits explicit rather than writing files.
    print("*** Begin Patch")
    for path, content in ((target, latex), (PAPER / "results_tables.md", markdown)):
        before = path.read_text().splitlines(keepends=True)
        after = content.splitlines(keepends=True)
        diff = list(difflib.unified_diff(before, after, n=3))[2:]
        if diff:
            print(f"*** Update File: {path}")
            for line in diff:
                print("@@" if line.startswith("@@") else line.rstrip("\n"))
    print("*** End Patch")
