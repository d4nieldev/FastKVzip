"""Shared, dependency-free benchmark names for evaluation, results and submission."""

RULER_TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
)
RULER_LENGTHS = {
    "4k": 4096,
    "8k": 8192,
    "16k": 16384,
    "32k": 32768,
    "64k": 65536,
    "128k": 131072,
}

SHORT = ("squad", "gsm")
MID = (
    "scbench_many_shot",
    "scbench_mf",
    "scbench_choice_eng",
    "scbench_qa_eng",
    "scbench_repoqa",
)
LONG = ("scbench_kv", "scbench_prefix_suffix", "scbench_summary", "scbench_vt")
MULTI = ("scbench_summary_with_needles", "scbench_repoqa_and_kv")
SCBENCH = MID + LONG + MULTI
# These are the sixteen additional files published in SCBench-preprocessed.
SCBENCH_VARIANTS = tuple(
    f"scbench_{task}_{length}"
    for task, lengths in (
        ("kv", ("tiny", "short", "mid")),
        ("many_shot", ("tiny", "short")),
        ("mf", ("tiny", "short", "mid")),
        ("prefix_suffix", ("tiny", "short", "mid")),
        ("repoqa", ("tiny", "short")),
        ("summary", ("tiny", "short", "mid")),
    )
    for length in lengths
)
RULER = tuple(
    f"ruler_{task}_{length}" for length in RULER_LENGTHS for task in RULER_TASKS
)
ALL_BENCHMARKS = SCBENCH + SCBENCH_VARIANTS + SHORT + RULER


class BenchmarkDataset(list):
    """Loaded rows plus the complete benchmark size, even for a limited pilot."""

    def __init__(self, rows, *, full_size):
        super().__init__(rows)
        self.full_size = full_size


def parse_ruler_name(name):
    """Return the task and length label of a concrete RULER dataset name."""
    task, separator, length = name.removeprefix("ruler_").rpartition("_")
    if (
        not name.startswith("ruler_")
        or not separator
        or task not in RULER_TASKS
        or length not in RULER_LENGTHS
    ):
        raise ValueError(f"Invalid RULER dataset: {name}")
    return task, length


def get_data_list(dataname, modelname=""):
    """Expand selectors without changing an explicitly requested model's datasets."""
    groups = {
        "short": SHORT,
        "mid": MID,
        "long": LONG,
        "multi": MULTI,
        "qa": SHORT + ("scbench_choice_eng", "scbench_qa_eng"),
        "retv": ("scbench_kv", "scbench_prefix_suffix", "scbench_repoqa"),
        "redun": ("scbench_summary", "scbench_vt", "scbench_mf", "scbench_many_shot"),
        "all": ALL_BENCHMARKS,
        "ruler": RULER,
    }
    if dataname in groups:
        return list(groups[dataname])
    if dataname.startswith("ruler_") and dataname[6:] in RULER_LENGTHS:
        return [name for name in RULER if name.endswith(f"_{dataname[6:]}")]
    if dataname.startswith("ruler_"):
        parse_ruler_name(dataname)
    return [dataname]
