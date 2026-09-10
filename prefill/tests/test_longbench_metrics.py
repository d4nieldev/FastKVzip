"""Golden cases for upstream LongBench revision 2e00731f8d0bff23dc4325161044d0ed8af94c1e."""

import pytest

from results.evaluation_run import EvaluationRun
from results.metric import evaluate_answer
from results.parse import build_run_metrics, parse_answer


METRIC_CASES = [
    ("narrativeqa", "The red fox!", ["blue fox", "red fox"], [], 1),
    ("qasper", "red blue", ["red green"], [], 0.5),
    ("multifieldqa_en", "one", ["1"], [], 0),
    ("multifieldqa_zh", "北京。", ["北京"], [], 1),
    ("hotpotqa", "red blue", ["red"], [], 2 / 3),
    ("2wikimqa", "The answer!", ["answer"], [], 1),
    ("musique", "red blue", ["green", "blue"], [], 2 / 3),
    ("dureader", "北京", ["上海", "北京"], [], 1),
    ("gov_report", "red blue", ["red blue"], [], 1),
    ("qmsum", "red blue", ["green", "red blue"], [], 1),
    ("multi_news", "red blue", ["red blue"], [], 1),
    ("vcsum", "北京", ["北京"], [], 1),
    ("trec", "\nHUM NUM\nDESC", ["HUM"], ["HUM", "NUM", "DESC"], 0.5),
    ("triviaqa", "\nred\nblue", ["blue", "red"], [], 1),
    ("samsum", "\nred blue\nwrong", ["red blue"], [], 1),
    ("lsht", "\n体育\n财经", ["体育"], ["体育", "财经"], 1),
    ("passage_count", "2 3 2", ["2"], [], 2 / 3),
    ("passage_retrieval_en", "Paragraph 12 or 2", ["Paragraph 12"], [], 0.5),
    ("passage_retrieval_zh", "段落12 或 2", ["段落12"], [], 0.5),
    ("lcc", "\n```python\n# comment\nreturn 1\nreturn 2", ["return 1"], [], 1),
    ("repobench-p", "// comment\nreturn 2", ["return 1", "return 2"], [], 1),
]


@pytest.mark.parametrize(
    "task,prediction,answers,classes,expected", METRIC_CASES,
)
def test_official_task_metrics_and_maximum_across_references(
    task, prediction, answers, classes, expected
):
    references = [answers]
    if task in {"trec", "lsht"}:
        references = {"answers": references, "all_classes": classes}
    assert evaluate_answer(
        [prediction], references, f"longbench_{task}", "qa"
    ) == [pytest.approx(expected, abs=1e-7)]


def test_all_original_tasks_have_golden_metric_cases():
    from data.benchmarks import get_data_list

    assert {f"longbench_{case[0]}" for case in METRIC_CASES} == set(get_data_list("longbench"))


@pytest.mark.parametrize(
    "prediction,answer,classes,expected",
    [
        ("DESC:reason", "DESC:reason", ["DESC", "DESC:reason"], 1),
        ("HUM NUM", "HUM", ["HUM", "NUM"], 0.5),
        ("hum", "HUM", ["HUM", "NUM"], 0),
        # Upstream filters matches in place: preserve its nested-substring behavior.
        ("ABC", "ABC", ["AB", "A", "ABC"], 0.5),
    ],
)
def test_official_classification_ambiguity_and_substring_behavior(
    prediction, answer, classes, expected
):
    refs = {"answers": [[answer]], "all_classes": classes}
    assert evaluate_answer([prediction], refs, "longbench_trec", "qa") == [expected]


@pytest.mark.parametrize("task", ["triviaqa", "trec", "lsht", "samsum"])
def test_first_line_rule_does_not_take_a_correct_answer_from_later_lines(task):
    refs = [["correct"]]
    if task in {"trec", "lsht"}:
        refs = {"answers": refs, "all_classes": ["correct", "wrong"]}
    assert evaluate_answer(["\nwrong\ncorrect"], refs, f"longbench_{task}", "qa") == [0]


def test_code_metric_keeps_indentation_and_uses_fuzzy_not_exact_match():
    from fuzzywuzzy import fuzz

    prediction = "    return 2"
    target = "    return 1"
    assert evaluate_answer(
        [f"# ignored\n{prediction}\n{target}"], [[target]], "longbench_lcc", "qa"
    ) == [fuzz.ratio(prediction, target) / 100]


@pytest.mark.parametrize(
    "task,target",
    [("passage_count", "2"), ("passage_retrieval_en", "Paragraph 2"), ("passage_retrieval_zh", "段落2")],
)
def test_count_and_retrieval_do_not_normalize_number_words(task, target):
    assert evaluate_answer(["two"], [[target]], f"longbench_{task}", "qa") == [0]


def test_longbench_dispatch_does_not_override_explicit_similarity():
    assert evaluate_answer(
        ["one"], ["1"], "longbench_qasper", "qa", similarity=True
    ) == [1]


def test_empty_predictions_and_reference_lists_score_zero():
    assert evaluate_answer(["", "answer"], [["answer"], []], "longbench_qasper", "qa") == [0, 0]


@pytest.mark.parametrize("task", ["longbench_trec", "longbench_lsht"])
def test_classification_supplement_is_full_ordered_and_shared_by_live_and_offline_scoring(
    monkeypatch, tmp_path, task
):
    from data import longbench

    rows = [
        {"answers": [["unused"]], "all_classes": ["unused"]},
        {"answers": [["DESC:reason"]], "all_classes": ["DESC", "DESC:reason"]},
        {"answers": [["NUM"]], "all_classes": ["NUM", "HUM"]},
    ]
    calls = []

    def load(name, *args, **kwargs):
        calls.append((name, args, kwargs))
        return rows

    monkeypatch.setattr(longbench, "load_longbench", load)
    assert parse_answer(task) == (rows, [])
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    with EvaluationRun.open(
        tmp_path, "run", checkpoint_path=checkpoint, wandb_run_id=None,
        window_size=0, level="pair",
    ) as run:
        for index, prediction in [(2, "NUM HUM"), (1, "DESC:reason")]:
            run.merge_example(task, index, outputs={"qa": [[
                [0.2, 0.2, 0.0, 0.2],
                {"pruned": prediction, "full__": rows[index]["answers"][0][0],
                 "answer": rows[index]["answers"][0]},
            ]]})
        live = build_run_metrics(run, dataset_sizes={task: 3})
    offline = build_run_metrics(EvaluationRun.load(tmp_path / "run"), dataset_sizes={task: 3})
    assert live == offline
    assert live["tasks"][task]["ratios"]["0.2"]["score"] == 75
    assert live["tasks"][task]["full_cache"]["score"] == 100
    assert not live["tasks"][task]["complete"]
    assert all(name == task and not args and not kwargs for name, args, kwargs in calls)
