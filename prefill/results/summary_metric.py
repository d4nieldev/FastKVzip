"""Assignment-matched ROUGE for full-cache and pruned summaries."""

from rouge_score.rouge_scorer import RougeScorer
from scipy.optimize import linear_sum_assignment


def matched_rouge(full: list[str], pruned: list[str]) -> dict:
    if not full or len(full) != len(pruned):
        raise ValueError("summary lists must have equal nonzero length")

    scorer = RougeScorer(("rougeL", "rouge1", "rouge2"), use_stemmer=False)
    scores = [
        [scorer.score(reference, candidate) for candidate in pruned]
        for reference in full
    ]
    rows, columns = linear_sum_assignment(
        [[entry["rougeL"].fmeasure for entry in row] for row in scores],
        maximize=True,
    )
    assignment = [[int(row), int(column)] for row, column in zip(rows, columns)]
    return {
        metric: sum(
            scores[row][column][metric].fmeasure
            for row, column in zip(rows, columns)
        )
        / len(full)
        for metric in ("rougeL", "rouge1", "rouge2")
    } | {"assignment": assignment}
