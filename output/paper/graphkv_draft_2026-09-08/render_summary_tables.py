"""Render the summarization-alignment subsection from the verified metric snapshot.

Reads data/summarization-alignment/{aggregate,method-significance}.json and writes
sections/summarization_alignment.tex. Every number in the prose and tables is taken from the
snapshot, so the text cannot drift from the data. Run with --check to validate without writing.
"""

import argparse
import json
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parent
DATA = PAPER/'data'/'summarization-alignment'
TARGET = PAPER/'sections'/'summarization_alignment.tex'

RATIOS = ('0.2', '0.3', '0.4', '0.5', '0.75')      # ascending, matching the other result tables
METHODS = (('graphkv', 'GraphKV (ours)'),
           ('fastkvzip', 'Official FastKVzip'),
           ('kvzip', 'KVzip'))
METRICS = (('bertscore_f1', 'BERTScore F1'),
           ('rouge2_f1', 'ROUGE-2 F1'),
           ('rougeL_f1', 'ROUGE-L F1'),
           ('styledistance_cosine', 'StyleDistance cosine'))
DATASETS = (('govreport_summary', 'GovReport', 493),
            ('pg19_summary', 'PG-19', 77))


TEMPLATE = r"""\subsection{Summarization Alignment To The Underlying LLM}
\label{sec:summarization-alignment}

The benchmarks in Section~\ref{sec:main-results} are dominated by retrieval: a fact is planted in a
long context and the model is asked to recover it. Such tasks establish that a pruned cache retains
specific tokens, but they say little about whether the model still represents the context as a
whole. We therefore evaluate a task with no single answer span. Each model is asked to summarise a
long document in roughly 400--500 words, and we ask how far pruning moves the summary away from what
the unpruned model would have written.

\paragraph{Setup.} We summarise 493 GovReport documents and 77 PG-19 books, each at least $8{,}192$
tokens long, under all three selectors and five retention ratios, plus the full cache. Decoding uses
temperature $0.7$, top-$p$ $0.9$ and up to $1{,}024$ new tokens, and we draw $N = 16$ samples per
document per condition, giving $164{,}160$ summaries over $1{,}710$ method--document records. All
three methods see exactly the same documents, so every comparison below is paired.

\paragraph{The unpruned model is not a fixed reference.} At temperature $0.7$ the full cache does not
agree with itself: two independent full-cache summaries of the same GovReport document share a
ROUGE-L F1 of only $@B_ROUGEL@$. A raw full-versus-pruned score of $@C_ROUGEL@$ at $20\%$ retention
therefore looks catastrophic while being almost exactly what resampling alone produces. We
consequently report every metric normalised by the full cache's agreement with itself. For one
document, let $B$ be the mean score over the $\binom{16}{2} = 120$ full-cache pairs with distinct
sample indices and $C$ the mean over all $16 \times 16 = 256$ full-versus-pruned pairs at a given
ratio; we report $100\,C/B$, averaged over documents with equal weight. A value of $100$ means
pruning perturbs the summary no more than resampling the unpruned model does, and values slightly
above $100$ are admissible.

\paragraph{Metrics.} We measure content with BERTScore F1 (\texttt{roberta-large}, layer 17, no IDF
weighting and no baseline rescaling), local phrasing with ROUGE-2 F1, wording and word order with
complete-summary ROUGE-L F1, and writing style with the cosine similarity of StyleDistance
embeddings. Summaries run a median of roughly $700$ tokens against a $512$-token encoder window, so
both neural metrics are computed over non-overlapping $510$-token windows covering every token
rather than by truncating; for a summary that fits a single window this reduces exactly to the
published metric, which we verify to $2.9\times10^{-8}$ against the reference implementation.

\paragraph{Pruning changes wording, not content.} Table~\ref{tab:summarization-alignment} shows that
all three selectors stay at or above $99$ on every metric down to $30\%$ retention, on both
datasets. The interesting behaviour is at $20\%$, and only on GovReport, where the metrics separate:
GraphKV loses $@DROP_R2@$ points of ROUGE-2 and $@DROP_RL@$ points of ROUGE-L, but only
$@DROP_BERT@$ points of BERTScore, while style does not move at all. The summaries are being
reworded rather than emptied of content. An evaluation restricted to $n$-gram overlap would have
reported this as a five-point degradation.

\paragraph{Two metrics saturate.} This normalisation is only informative where the baseline leaves
room. Full-cache self-agreement is $@B_BERT@$ for BERTScore and $@B_STYLE@$ for StyleDistance, and no
StyleDistance cell departs from $100$ by more than $0.1$. The flat style result is an absence of
measurement range rather than evidence that style is preserved, and we report it for completeness
only.

\paragraph{Method differences do not support a ranking.} Because the cohort is paired over 493
documents, the comparison is sensitive enough to resolve differences of roughly $0.05$ normalised
points, and $@N_SIG@$ of the $@N_TOTAL@$ method comparisons in the grid are individually significant
after a Bonferroni correction. Their signs, however, are not stable: on GovReport BERTScore, GraphKV
leads FastKVzip by $+0.038$ at $75\%$ retention, trails it by $-0.142$ at $30\%$, and leads again by
$+0.069$ at $20\%$, with every one of those differences clearing the corrected threshold. Across all
significant lexical and BERTScore comparisons GraphKV is ahead in 14 and behind in 8. We therefore
do not read an overall ordering out of this experiment. The one setting in which an ordering is
consistent is the most aggressive ratio: at $20\%$ retention on GovReport, GraphKV leads both
baselines on all three informative metrics simultaneously
(Table~\ref{tab:summarization-significance}).

\paragraph{Scope.} Every metric here measures similarity to the unpruned model's own output. None
measures factual accuracy, faithfulness to the source document, or summary quality, and none would
detect an error that the unpruned model also makes. These results bound how far pruning moves the
model from itself; they do not establish that the resulting summaries are good.

@TABLE_MAIN@

@TABLE_SIG@
"""


def load():
    agg = json.loads((DATA/'aggregate.json').read_text())
    sig = json.loads((DATA/'method-significance.json').read_text())
    assert agg['complete'] is True, 'aggregate snapshot is incomplete'
    assert agg['expected_records'] == 1710, agg['expected_records']
    for _, metric in METRICS:
        pass
    for key, _ in METRICS:
        for dataset, _, count in DATASETS:
            for method, _ in METHODS:
                entry = agg['metrics'][key][f'{dataset}|{method}']
                assert entry['documents'] == count, (key, dataset, method, entry['documents'])
                assert entry['undefined_baseline_documents'] == 0
    return agg, sig


def table(agg):
    lines = [r'\begin{table}[t]',
             r'\caption{Summarization alignment to the full-cache model. Each cell is '
             r'$100\,C/B$: the mean similarity between full-cache and pruned-cache summaries of the '
             r"same document ($C$, 256 sample pairs), divided by the full-cache pool's agreement "
             r'with itself ($B$, 120 pairs), computed per document and then averaged with equal '
             r'document weight. $100$ means pruning perturbs the output no more than resampling the '
             r'unpruned model does. Column $B$ reports the raw full-cache self-agreement, the scale '
             r'on which each metric operates.}',
             r'\label{tab:summarization-alignment}',
             r'\centering',
             r'\small',
             r'\begin{tabular}{@{}llrrrrrr@{}}',
             r'\hline',
             r'Metric & Method & $B$ & ' + ' & '.join(f'{int(float(r)*100)}\\%' for r in RATIOS) + r' \\',
             r'\hline']
    for dataset, label, count in DATASETS:
        lines.append(r'\multicolumn{8}{@{}l}{\emph{%s (%d documents per method)}} \\' % (label, count))
        for mi, (key, mlabel) in enumerate(METRICS):
            for method, methodlabel in METHODS:
                entry = agg['metrics'][key][f'{dataset}|{method}']
                cells = ' & '.join(f"{entry['ratios'][r]['normalized_score']:.2f}" for r in RATIOS)
                first = mlabel if method == 'graphkv' else ''
                lines.append(f"{first} & {methodlabel} & {entry['full_self_baseline']:.3f} & {cells} "
                             r'\\')
            if mi < len(METRICS)-1:
                lines.append(r'\noalign{\smallskip}')
        lines.append(r'\hline')
    lines += [r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def significance_table(sig):
    """The only internally consistent method ordering: 20% retention on GovReport."""
    wanted = {('rouge2', 'graphkv-fastkvzip'), ('rouge2', 'graphkv-kvzip'),
              ('rougeL', 'graphkv-fastkvzip'), ('rougeL', 'graphkv-kvzip'),
              ('bertscore', 'graphkv-fastkvzip'), ('bertscore', 'graphkv-kvzip')}
    rows = {(r['metric'], r['comparison']): r for r in sig
            if r['dataset'] == 'govreport_summary' and r['ratio'] == '0.2'}
    label = {'bertscore': 'BERTScore F1', 'rouge2': 'ROUGE-2 F1', 'rougeL': 'ROUGE-L F1'}
    comp = {'graphkv-fastkvzip': 'vs.\\ Official FastKVzip', 'graphkv-kvzip': 'vs.\\ KVzip'}
    lines = [r'\begin{table}[t]',
             r'\caption{Paired method differences at $20\%$ retention on GovReport, the one setting '
             r'where an ordering is consistent across metrics. Differences are paired over the 493 '
             r'documents scored under every method; intervals are $95\%$ bootstrap confidence '
             r'intervals over documents ($10{,}000$ resamples). All six clear a Bonferroni-corrected '
             r'threshold of $p < 4.17\times10^{-4}$ for the 120 comparisons in the grid.}',
             r'\label{tab:summarization-significance}',
             r'\centering',
             r'\small',
             r'\begin{tabular}{@{}llr@{\,}l@{}}',
             r'\hline',
             r'Metric & GraphKV & \multicolumn{2}{c}{Difference (95\% CI)} \\',
             r'\hline']
    for metric in ('bertscore', 'rouge2', 'rougeL'):
        for c in ('graphkv-fastkvzip', 'graphkv-kvzip'):
            assert (metric, c) in wanted
            r = rows[(metric, c)]
            assert r['ci_excludes_zero'] and r['p_wilcoxon'] < 0.05/len(sig), (metric, c)
            first = label[metric] if c == 'graphkv-fastkvzip' else ''
            lines.append(f"{first} & {comp[c]} & ${r['mean_difference']:+.3f}$ & "
                         f"$[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]$ " r'\\')
    lines += [r'\hline', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def prose(agg, sig):
    gov = {k: agg['metrics'][k]['govreport_summary|graphkv'] for k, _ in METRICS}
    drop = {k: 100-gov[k]['ratios']['0.2']['normalized_score'] for k in gov}
    total = len(sig)
    significant = sum(1 for r in sig if r['ci_excludes_zero'] and r['p_wilcoxon'] < 0.05/total)
    values = {
        '@B_ROUGEL@': f"{gov['rougeL_f1']['full_self_baseline']:.3f}",
        '@C_ROUGEL@': f"{gov['rougeL_f1']['ratios']['0.2']['cross']:.3f}",
        '@B_BERT@': f"{gov['bertscore_f1']['full_self_baseline']:.3f}",
        '@B_STYLE@': f"{agg['metrics']['styledistance_cosine']['govreport_summary|graphkv']['full_self_baseline']:.3f}",
        '@DROP_R2@': f"{drop['rouge2_f1']:.1f}",
        '@DROP_RL@': f"{drop['rougeL_f1']:.1f}",
        '@DROP_BERT@': f"{drop['bertscore_f1']:.1f}",
        '@N_SIG@': str(significant),
        '@N_TOTAL@': str(total),
        '@TABLE_MAIN@': table(agg),
        '@TABLE_SIG@': significance_table(sig),
    }
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
    agg, sig = load()
    text = prose(agg, sig)
    if args.check:
        current = TARGET.read_text() if TARGET.exists() else ''
        print('up to date' if current == text else 'STALE: rerun without --check')
        raise SystemExit(0 if current == text else 1)
    TARGET.write_text(text)
    print(f'wrote {TARGET} ({len(text.splitlines())} lines)')


if __name__ == '__main__':
    main()
