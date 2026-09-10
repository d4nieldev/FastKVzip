import pytest

from results.summary_metric import matched_rouge


def test_assignment_maximizes_full_rouge_l_and_reuses_it_for_all_scores():
    result = matched_rouge(
        ["alpha beta", "gamma delta"],
        ["gamma delta", "alpha beta"],
    )

    assert result == {
        "rougeL": 1.0,
        "rouge1": 1.0,
        "rouge2": 1.0,
        "assignment": [[0, 1], [1, 0]],
    }


def test_assignment_preserves_duplicate_outputs():
    result = matched_rouge(["same", "same"], ["same", "different"])

    assert len(result["assignment"]) == 2
    assert sorted(pair[0] for pair in result["assignment"]) == [0, 1]
    assert sorted(pair[1] for pair in result["assignment"]) == [0, 1]
    assert result["rougeL"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "full,pruned",
    [([], []), (["one"], []), (["one"], ["one", "two"])],
)
def test_assignment_requires_equal_nonempty_output_lists(full, pruned):
    with pytest.raises(ValueError, match="equal nonzero"):
        matched_rouge(full, pruned)
