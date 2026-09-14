"""Render the LongBench v2 paragraph and table from the verified score snapshot.

Reads data/longbench-v2-scores.json and writes sections/longbench_v2.tex. Every number in the
prose and table is taken from the snapshot, so the text cannot drift from the data. Run with
--check to validate without writing.
"""

import argparse
import json
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parent
DATA = PAPER/'data'/'longbench-v2-scores.json'
TARGET = PAPER/'sections'/'longbench_v2.tex'

RATIOS = ('0.2', '0.3', '0.4', '0.5', '0.75')      # ascending, matching the other result tables
METHODS = (('graphkv', 'GraphKV (ours)'),
           ('fastkvzip', 'Official FastKVzip'),
           ('kvzip', 'KVzip'))
SHORT = {'graphkv': 'GraphKV', 'fastkvzip': 'Fast KVzip', 'kvzip': 'KVzip'}

TEMPLATE = r"""\paragraph{LongBench v2.}
The benchmarks above are dominated by retrieval. LongBench v2
\citep{bai2024longbenchv2} instead asks @N@ four-way multiple-choice questions
that require understanding and reasoning over realistic long contexts:
single- and multi-document QA, long in-context learning, long dialogue
history, code repositories, and structured data, with contexts from 8K to
2M words. We evaluate all @N@ questions with the same checkpoint, protected
windows, and chunking as above, truncate contexts to the model's 1M-token
limit (the longest raw context has @LONGEST_M@M tokens), ask for the answer
directly without chain-of-thought, and decode greedily for up to @MAX_NEW@
tokens. The score is accuracy on a 0--100 scale, so one question is
@ONE_Q@ points. The unpruned model reaches @FULL@ (@FULL_CORRECT@/@N@), only
@ABOVE_CHANCE@ points above the 25-point chance level, which leaves little
room for pruning to hurt. Every pruned configuration in
Table~\ref{tab:longbench-v2} lies within @MAX_DROP@ points of the full cache,
the largest gap between methods at any ratio is @MAX_GAP@ points, and no
method leads at every ratio (@LEADERS@). The clearest separation is at
20\% retention, where GraphKV is @GK_20_DROP@ points below the full cache
while Fast KVzip loses @FK_20_DROP@ and KVzip @KV_20_DROP@ points; across the
five ratios GraphKV's scores span @GK_SPAN@ points, Fast KVzip's @FK_SPAN@,
and KVzip's @KV_SPAN@. These differences amount to a handful of questions,
and we do not read a ranking from them. What this benchmark shows is that
aggressive pruning leaves long-context multiple-choice accuracy essentially
unchanged for all three selectors.

\begin{table}[t]
\caption{LongBench v2 accuracy (0--100) on all @N@ questions at each retention ratio; the last column is the unpruned model. One question is @ONE_Q@ points.}
\label{tab:longbench-v2}
\centering
\small
\begin{tabular}{@{}lrrrrrr@{}}
\hline
Method & 20\% & 30\% & 40\% & 50\% & 75\% & Full \\
\hline
@ROWS@
\hline
\end{tabular}
\end{table}
"""


def load():
    return json.loads(DATA.read_text())


def f2(x):
    return f'{x:.2f}'


def f1(x):
    return f'{x:.1f}'


def leaders_text(scores):
    """'GraphKV at 20\\%, 30\\%, and 75\\%; KVzip at 40\\% and 50\\%' with ties listed jointly."""
    by_method = {}
    for r in RATIOS:
        best = max(scores[m][r] for m, _ in METHODS)
        winners = [SHORT[m] for m, _ in METHODS if abs(scores[m][r] - best) < 1e-9]
        by_method.setdefault(' and '.join(winners), []).append(r)
    parts = []
    for who, ratios in by_method.items():
        labels = [f'{int(round(float(r) * 100))}\\%' for r in ratios]
        if len(labels) == 1:
            joined = labels[0]
        elif len(labels) == 2:
            joined = ' and '.join(labels)
        else:
            joined = ', '.join(labels[:-1]) + ', and ' + labels[-1]
        parts.append(f'{who} at {joined}')
    return '; '.join(parts)


def prose(snap):
    scores = snap['scores']
    n = snap['benchmark']['questions']
    full = {m: scores[m]['1'] for m, _ in METHODS}
    assert max(full.values()) - min(full.values()) < 1e-9, 'full-cache scores differ between methods'
    full_score = full['graphkv']
    for m, _ in METHODS:
        for r in RATIOS + ('1',):
            c = scores[m][r] * n / 100
            assert abs(c - round(c)) < 1e-6, (m, r, scores[m][r])
    drops = {m: {r: scores[m][r] - full_score for r in RATIOS} for m, _ in METHODS}
    max_drop = max(-d for m in drops for d in drops[m].values())
    max_gap = max(max(scores[m][r] for m, _ in METHODS) - min(scores[m][r] for m, _ in METHODS) for r in RATIOS)
    span = {m: max(scores[m][r] for r in RATIOS) - min(scores[m][r] for r in RATIOS) for m, _ in METHODS}
    rows = []
    for m, label in METHODS:
        cells = [f2(scores[m][r]) for r in RATIOS] + [f2(scores[m]['1'])]
        rows.append(f'{label} & ' + ' & '.join(cells) + r' \\')
    values = {
        '@N@': str(n),
        '@LONGEST_M@': f1(snap['benchmark']['longest_raw_context_tokens'] / 1e6),
        '@MAX_NEW@': str(snap['benchmark']['max_new_tokens']),
        '@ONE_Q@': f1(100 / n),
        '@FULL@': f2(full_score),
        '@FULL_CORRECT@': str(int(round(full_score * n / 100))),
        '@ABOVE_CHANCE@': f1(full_score - snap['benchmark']['chance_level']),
        '@MAX_DROP@': f1(max_drop),
        '@MAX_GAP@': f1(max_gap),
        '@LEADERS@': leaders_text(scores),
        '@GK_20_DROP@': f1(-drops['graphkv']['0.2']),
        '@FK_20_DROP@': f1(-drops['fastkvzip']['0.2']),
        '@KV_20_DROP@': f1(-drops['kvzip']['0.2']),
        '@GK_SPAN@': f1(span['graphkv']),
        '@FK_SPAN@': f1(span['fastkvzip']),
        '@KV_SPAN@': f1(span['kvzip']),
        '@ROWS@': '\n'.join(rows),
    }
    assert all(-d >= 0 for d in (drops['graphkv']['0.2'], drops['fastkvzip']['0.2'], drops['kvzip']['0.2'])), \
        'prose says "below the full cache" at 20%; the data no longer supports that wording'
    text = TEMPLATE
    for key, value in values.items():
        text = text.replace(key, value)
    leftover = re.findall(r'@[A-Z_0-9]+@', text)
    assert not leftover, f'unsubstituted placeholders: {leftover}'
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='validate without writing')
    args = ap.parse_args()
    text = prose(load())
    if args.check:
        current = TARGET.read_text() if TARGET.exists() else ''
        print('up to date' if current == text else 'STALE: rerun without --check')
        raise SystemExit(0 if current == text else 1)
    TARGET.write_text(text)
    print(f'wrote {TARGET} ({len(text.splitlines())} lines)')


if __name__ == '__main__':
    main()
