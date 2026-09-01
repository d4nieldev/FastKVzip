import json
from dataclasses import dataclass

import numpy as np
from datasets import Dataset, load_dataset
from tqdm import tqdm


@dataclass(frozen=True)
class FineWebTrainingData:
    datasets: dict[str, dict[int, dict]]
    train_keys: tuple[tuple[str, int], ...]
    validation_keys: tuple[tuple[str, int], ...]


def load_dataset_all(name, tokenizer, n_data=100):
    """
    Each data example has a format of {context: str, question: List[str], answers: List[str]}.

    possible datasets = ["squad", "gsm",
                        ""scbench_kv", "scbench_vt",  scbench_many_shot", "scbench_mf", "scbench_repoqa",
                        "scbench_choice_eng", "scbench_prefix_suffix", "scbench_summary", "scbench_qa_eng",
                        "scbench_summary_with_needles", "scbench_repoqa_and_kv"]

    Note:
        We preprocess SCBench to follow the data format described above.
        Additionally, we subsample scbench_choice_eng and scbench_qa_eng to ensure that the context token length (LLaMA3 tokenizer)
        is less than 125K, fitting within the context limit of LLaMA3 models.
        These preprocessed datasets are available on Hugging Face: Jang-Hyun/SCBench-preprocessed

        We also provide shortened SCBench, excluding tasks {choice_eng, qa_eng, vt}, which are difficult to shorten.
        - The "tiny" tag (e.g., scbench_kv_tiny) has a context length of approximately 8k tokens.
        - The "short" tag (e.g., scbench_kv_short) has a context length of approximately 20k tokens.
    """

    if name == "squad":
        dataset = load_squad(n_data)
    elif name == "gsm":
        dataset = load_gsm(tokenizer, n_data)
    elif "scbench" in name:
        dataset = load_scbench(name)
    elif "fineweb" in name:
        dataset = load_fineweb(name)
    elif "mrcr" in name:
        dataset = load_mrcr(tokenizer, n_data)
    else:
        raise ValueError(f"Invalid dataset: {name}")

    print(f"\n{name} loaded, #data: {len(dataset)}")
    return dataset


def load_squad(n_data):
    data = load_dataset("rajpurkar/squad", split="train")

    pool = dict()
    dataset = {"context": [], "question": [], "answers": []}
    for d in data:
        # aggregate qa pairs for the shared context
        if d["context"] not in pool:
            pool[d["context"]] = len(dataset["context"])
            dataset["context"].append(d["context"])
            dataset["question"].append([d["question"]])
            dataset["answers"].append(d["answers"]["text"])
        else:
            idx = pool[d["context"]]
            assert dataset["context"][idx] == d["context"]
            dataset["question"][idx].append(d["question"])
            dataset["answers"][idx].append(d["answers"]["text"][0])

        if len(pool) > n_data:
            break

    dataset = Dataset.from_dict(dataset)
    return dataset


def load_gsm(tokenizer, n_data):
    dataset_full = load_dataset("openai/gsm8k", "main", split="test")

    dataset = []
    for data in dataset_full:
        st = data["question"].split(". ")

        data["context"] = ". ".join(st[:-1]).strip() + "."
        l = len(tokenizer.encode(data["context"], add_special_tokens=False))
        if l < 72:  # pass short context
            continue

        data["question"] = [st[-1].strip()]
        data["answers"] = [data["answer"]]
        dataset.append(data)

        if len(dataset) == n_data:
            break

    return dataset


def load_scbench(name):
    check_scbench_name(name)
    samples = load_dataset(
        "Jang-Hyun/SCBench-preprocessed",
        data_files=f"{name}.parquet",
        split="train",
    )

    dataset = []
    for data in samples:
        d = {}
        d["context"] = data["prompts"][0]
        d["question"] = data["prompts"][1:]
        d["answers"] = []
        for gt in data["ground_truth"]:
            if isinstance(gt, list):
                gt = ", ".join(gt)
            else:
                gt = str(gt)
            d["answers"].append(gt)

        dataset.append(d)

    return dataset


def check_scbench_name(name):
    name = name.split("scbench_")[1]
    possible_tags = [
        "many_shot",
        "mf",
        "repoqa",
        "choice_eng",
        "prefix_suffix",
        "summary",
        "qa_eng",
        "vt",
        "kv",
        "summary_with_needles",
        "repoqa_and_kv",
    ]
    if "tiny" in name:
        name = name.split("_tiny")[0]
    elif "short" in name:
        name = name.split("_short")[0]
    elif "mid" in name:
        name = name.split("_mid")[0]

    assert name in possible_tags, "SCBench data name not exist!"


def load_fineweb(name):
    """fineweb-[10k, 10k-cat, 100k]"""

    samples, length = _load_fineweb_source()

    if "10k" in name:
        min_len, max_len = 10000, 30000
    elif "100k" in name:
        min_len, max_len = 100000, 125000
    else:
        raise AssertionError("check fineweb dataset name!")

    valid = _valid_fineweb_indices(length, min_len, max_len)
    if "cat" in name:
        dataset, _, _ = _select_fineweb_concat(
            samples, valid, token_target=1_000_001
        )
    else:
        dataset, _, _ = _select_fineweb_regular(
            samples, valid, token_target=1_000_001
        )
    return list(dataset.values())


def _load_fineweb_source():
    samples = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        data_files="sample/10BT/000_00000.parquet",
        split="train",
    )
    return samples, np.array(samples.data.column("token_count"))


def _valid_fineweb_indices(length, min_len=10_000, max_len=30_000):
    return np.flatnonzero((length >= min_len) & (length < max_len)).tolist()


def _fineweb_example(text):
    return {"context": text, "question": [""], "answers": [""]}


def _select_fineweb_regular(
    samples,
    valid,
    *,
    start_source_index=0,
    count=None,
    token_target=None,
):
    if (count is None) == (token_target is None):
        raise ValueError("select regular FineWeb data by count or token target")

    dataset = {}
    total = 0
    last_source_index = None
    for source_index in valid:
        source_index = int(source_index)
        if source_index < start_source_index:
            continue
        sample = samples[source_index]
        dataset[source_index] = _fineweb_example(sample["text"].strip())
        total += int(sample["token_count"])
        last_source_index = source_index
        if (count is not None and len(dataset) == count) or (
            token_target is not None and total >= token_target
        ):
            break

    if (count is not None and len(dataset) < count) or (
        token_target is not None and total < token_target
    ):
        raise ValueError("FineWeb source does not contain enough regular contexts")
    return dataset, total, last_source_index


def _select_fineweb_concat(
    samples,
    valid,
    *,
    start_source_index=0,
    count=None,
    token_target=None,
):
    if (count is None) == (token_target is None):
        raise ValueError("select concatenated FineWeb data by count or token target")

    dataset = {}
    total = 0
    group_start = None
    group_text = ""
    group_tokens = 0
    last_source_index = None
    for source_index in valid:
        source_index = int(source_index)
        if source_index < start_source_index:
            continue
        sample = samples[source_index]
        if group_start is None:
            group_start = source_index
        group_text += "\n\n" + sample["text"].strip()
        group_tokens += int(sample["token_count"])
        last_source_index = source_index
        if group_tokens < 100_000:
            continue

        dataset[group_start] = _fineweb_example(group_text)
        total += group_tokens
        group_start = None
        group_text = ""
        group_tokens = 0
        if (count is not None and len(dataset) == count) or (
            token_target is not None and total >= token_target
        ):
            break

    if (count is not None and len(dataset) < count) or (
        token_target is not None and total < token_target
    ):
        raise ValueError("FineWeb source does not contain enough concatenated contexts")
    return dataset, total, last_source_index


def load_fineweb_training(train_context_count=29):
    """Build independent regular/concatenated training pools and held-out data."""

    if train_context_count < 1:
        raise ValueError("train context count must be positive")
    samples, length = _load_fineweb_source()
    valid = _valid_fineweb_indices(length)

    regular_train, token_target, regular_last = _select_fineweb_regular(
        samples, valid, count=train_context_count
    )
    concat_train, _, concat_last = _select_fineweb_concat(
        samples, valid, token_target=token_target
    )
    last_training_source_index = max(regular_last, concat_last)
    validation_start = last_training_source_index + 1
    regular_validation, _, _ = _select_fineweb_regular(
        samples, valid, start_source_index=validation_start, count=3
    )
    concat_validation, _, _ = _select_fineweb_concat(
        samples, valid, start_source_index=validation_start, count=1
    )

    regular = {**regular_train, **regular_validation}
    concat = {**concat_train, **concat_validation}
    return FineWebTrainingData(
        datasets={"fineweb_10k": regular, "fineweb_10k_cat": concat},
        train_keys=tuple(
            [("fineweb_10k", index) for index in regular_train]
            + [("fineweb_10k_cat", index) for index in concat_train]
        ),
        validation_keys=tuple(
            [("fineweb_10k", index) for index in regular_validation]
            + [("fineweb_10k_cat", index) for index in concat_validation]
        ),
    )


def build_prompt_text(sample):
    """Build prompt text from sample messages (mrcr)"""
    messages = json.loads(sample["prompt"])
    prompt_text = ""
    for msg in messages[:-1]:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            prompt_text += f"User: {content}\n\n"
        else:
            prompt_text += f"Assistant: {content}\n\n"
    return prompt_text, messages[-1]["content"]


def load_mrcr(tokenizer, n_data=2400, max_tokens=128000, n_needles=None):
    """Load MRCR dataset filtered by actual token count"""
    dataset = load_dataset("openai/mrcr", name="default")["train"]

    data_list = []
    print(f"Filtering samples by token count (max_tokens={max_tokens})...")

    for sample in tqdm(dataset, desc="Tokenizing"):
        if n_needles is not None and sample["n_needles"] != n_needles:
            continue

        prompt_text, last_query = build_prompt_text(sample)
        n_tokens = len(tokenizer.encode(prompt_text))

        if n_tokens <= max_tokens:
            sample_with_tokens = dict(sample)
            sample_with_tokens["n_tokens"] = n_tokens
            sample_with_tokens["prompt"] = prompt_text
            sample_with_tokens["query"] = last_query
            data_list.append(sample_with_tokens)

        if len(data_list) >= n_data:
            break

    return data_list


if __name__ == "__main__":
    import argparse

    from eval import get_data_list
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-m", "--model", type=str, default="Qwen/Qwen2.5-7B-Instruct-1M"
    )
    parser.add_argument("-d", "--data", type=str, help="check data/load.py for a list")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    for args.data in get_data_list(args.data):
        dataset = load_dataset_all(args.data, tokenizer)
        print(len(dataset))

        lengths = []
        for i, d in enumerate(dataset):
            print("=" * 50, "\n", d["context"][:140])
            l = len(tokenizer.encode(d["context"], add_special_tokens=False))
            lengths.append(l)
            print(i, sum(lengths))

            print(d["question"][0])

            break

        print()
        print(args.data, round(sum(lengths) / len(lengths), 0), max(lengths))
        print(lengths)
