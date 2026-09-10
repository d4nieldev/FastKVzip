"""Original LongBench test splits and official task text, with GraphKV wrapping."""

import json
import zipfile

from data.benchmarks import BenchmarkDataset, parse_longbench_name

LONGBENCH_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
LONGBENCH_PROTOCOL = "graphkv-v1"

# Task text and output caps from the MIT-licensed THUDM/LongBench repository:
# https://github.com/THUDM/LongBench/tree/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/config
# Copyright (c) 2023 THU-KEG & Zhipu AI; full MIT notice: results/longbench.py.
PROMPTS = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}
MAX_NEW_TOKENS = {
    "narrativeqa": 128, "qasper": 128, "multifieldqa_en": 64, "multifieldqa_zh": 64,
    "hotpotqa": 32, "2wikimqa": 32, "musique": 32, "dureader": 128,
    "gov_report": 512, "qmsum": 512, "multi_news": 512, "vcsum": 512,
    "trec": 64, "triviaqa": 32, "samsum": 128, "lsht": 64, "passage_count": 32,
    "passage_retrieval_en": 32, "passage_retrieval_zh": 32, "lcc": 64, "repobench-p": 64,
}


def parse_longbench_row(task, source):
    """Separate protected task text from the raw compressible context."""
    if task not in PROMPTS:
        raise ValueError(f"Invalid LongBench task: {task}")
    context, query = source.get("context"), source.get("input")
    answers, classes = source.get("answers"), source.get("all_classes", [])
    if classes is None:
        classes = []
    if (
        not isinstance(context, str)
        or not isinstance(query, str)
        or not isinstance(answers, list)
        or not answers
        or not all(isinstance(answer, str) for answer in answers)
        or not isinstance(classes, list)
        or not all(isinstance(label, str) for label in classes)
        or (task in ("trec", "lsht") and not classes)
    ):
        raise ValueError(f"Malformed LongBench {task} row: invalid text, answers or classes")
    prefix, suffix = PROMPTS[task].split("{context}")
    return {
        "context": context,
        "question": [suffix.format(input=query)],
        "answers": [answers],
        "context_prefix": prefix.format(input=query),
        "all_classes": classes,
    }


def load_longbench(name, n_data=None, *, start=0):
    """Read only this task's JSONL member from the pinned HF-cached archive."""
    from huggingface_hub import hf_hub_download

    task = parse_longbench_name(name)
    if start < 0 or (n_data is not None and n_data < 0):
        raise ValueError("LongBench range must be non-negative")
    path = hf_hub_download(
        "zai-org/LongBench", filename="data.zip", repo_type="dataset",
        revision=LONGBENCH_REVISION,
    )
    stop = None if n_data is None else start + n_data
    rows, full_size = [], 0
    with zipfile.ZipFile(path) as archive, archive.open(f"data/{task}.jsonl") as samples:
        for index, line in enumerate(samples):
            full_size += 1
            if index >= start and (stop is None or index < stop):
                rows.append(parse_longbench_row(task, json.loads(line)))
    return BenchmarkDataset(rows, full_size=full_size)
