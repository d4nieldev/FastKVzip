"""LongBench scoring, adapted from original metrics.py/eval.py and v2 pred.py at
https://github.com/THUDM/LongBench/tree/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench
https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/pred.py

MIT License
Copyright (c) 2023 THU-KEG & Zhipu AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import re
import string

from fuzzywuzzy import fuzz


def _qa_score(prediction, reference, *, chinese=False, **kwargs):
    from results.metric import f1_score

    def normalize(text):
        punctuation = string.punctuation
        if chinese:
            punctuation += "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        text = "".join(char for char in text.lower() if char not in punctuation)
        if chinese:
            return "".join(text.split())
        return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())

    if chinese:
        import jieba

        prediction = " ".join(
            filter(None, map(normalize, jieba.cut(prediction, cut_all=False)))
        )
        reference = " ".join(
            filter(None, map(normalize, jieba.cut(reference, cut_all=False)))
        )
    else:
        prediction, reference = normalize(prediction), normalize(reference)
    return f1_score(prediction, reference, normalize=False)


def _rouge_score(prediction, reference, *, chinese=False, **kwargs):
    from results.metric import rouge_score

    if chinese:
        import jieba

        prediction = " ".join(jieba.cut(prediction, cut_all=False))
        reference = " ".join(jieba.cut(reference, cut_all=False))
    return rouge_score(prediction, reference)


def _count_score(prediction, reference, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    return numbers.count(str(reference)) / len(numbers) if numbers else 0.0


def _retrieval_score(prediction, reference, *, chinese=False, **kwargs):
    pattern = r"段落(\d+)" if chinese else r"Paragraph (\d+)"
    return _count_score(prediction, re.findall(pattern, reference)[0])


def _classification_score(prediction, reference, *, all_classes, **kwargs):
    matches = [name for name in all_classes if name in prediction]
    # Preserve the official scorer's in-place substring filtering.
    for match in matches:
        if match in reference and match != reference:
            matches.remove(match)
    return 1 / len(matches) if reference in matches else 0.0


def _code_score(prediction, reference, **kwargs):
    prediction = next(
        (
            line for line in prediction.lstrip("\n").split("\n")
            if not any(marker in line for marker in ("`", "#", "//"))
        ),
        "",
    )
    return fuzz.ratio(prediction, reference) / 100


_METRICS = {
    "narrativeqa": _qa_score,
    "qasper": _qa_score,
    "multifieldqa_en": _qa_score,
    "multifieldqa_zh": _qa_score,
    "hotpotqa": _qa_score,
    "2wikimqa": _qa_score,
    "musique": _qa_score,
    "dureader": _rouge_score,
    "gov_report": _rouge_score,
    "qmsum": _rouge_score,
    "multi_news": _rouge_score,
    "vcsum": _rouge_score,
    "trec": _classification_score,
    "triviaqa": _qa_score,
    "samsum": _rouge_score,
    "lsht": _classification_score,
    "passage_count": _count_score,
    "passage_retrieval_en": _retrieval_score,
    "passage_retrieval_zh": _retrieval_score,
    "lcc": _code_score,
    "repobench-p": _code_score,
}


def evaluate_longbench(predictions, references, dataname):
    task = dataname.removeprefix("longbench_")
    metric = _METRICS[task]
    chinese = task in {"multifieldqa_zh", "dureader", "vcsum", "passage_retrieval_zh"}
    all_classes = None
    if task in {"trec", "lsht"}:
        all_classes = references["all_classes"]
        references = references["answers"]
    scores = []
    for prediction, answers in zip(predictions, references):
        if task in {"trec", "triviaqa", "samsum", "lsht"}:
            prediction = prediction.lstrip("\n").split("\n")[0]
        scores.append(
            max(
                (
                    metric(prediction, answer, all_classes=all_classes, chinese=chinese)
                    for answer in answers
                ),
                default=0.0,
            )
        )
    return scores


def evaluate_longbench_v2(predictions, references):
    scores = []
    for prediction, answers in zip(predictions, references):
        prediction = prediction.replace("*", "")
        match = re.search(r"The correct answer is \(([A-D])\)", prediction)
        if match is None:
            match = re.search(r"The correct answer is ([A-D])", prediction)
        scores.append(float(match is not None and match.group(1) in answers))
    return scores
