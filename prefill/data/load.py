import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from data.benchmarks import BenchmarkDataset
from generation import GENERATION_REVISION

AGENTIC_DATASET = "yzhuang/Agentic-Long-Context-Understanding-QA"


def available_splits(name):
    if name == "agentic":
        return frozenset({"train", "test"})
    return frozenset()


def load_dataset_all(
    name,
    tokenizer,
    n_data=100,
    *,
    split="test",
    teacher=None,
    answer_cache_dir=None,
    start=0,
    count=None,
):
    """
    Each example has context text and aligned question/answers lists. A RULER
    or LongBench answer is a list of targets/aliases for its single question. For fixed
    evaluation benchmarks, n_data=None loads the full benchmark; start/count
    select contexts, and the returned sequence records its full_size.

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

    if count is not None:
        n_data = count
    if start < 0 or (n_data is not None and n_data < 0):
        raise ValueError("Dataset range must be non-negative")
    if name == "agentic":
        dataset = load_agentic(
            split,
            teacher=teacher,
            answer_cache_dir=answer_cache_dir,
            start=start,
            count=n_data,
        )
    elif name == "squad":
        dataset = load_squad(n_data, start=start)
    elif name == "gsm":
        dataset = load_gsm(tokenizer, n_data, start=start)
    elif name.startswith("ruler_"):
        from data.ruler import load_ruler

        dataset = load_ruler(name, n_data, start=start)
    elif name.startswith("longbench_"):
        from data.longbench import load_longbench

        if split != "test":
            raise ValueError(f"Invalid LongBench split: {split}; only test is available")
        dataset = load_longbench(name, n_data, start=start)
    elif "scbench" in name:
        dataset = load_scbench(name, n_data, start=start)
    elif "fineweb" in name:
        dataset = load_fineweb(name)
    elif "mrcr" in name:
        dataset = load_mrcr(tokenizer, n_data, start=start)
    else:
        raise ValueError(f"Invalid dataset: {name}")

    print(f"\n{name} loaded, #data: {len(dataset)}")
    return dataset


def load_agentic(split, *, teacher, answer_cache_dir, start, count):
    if split not in available_splits("agentic"):
        raise ValueError(f"Invalid Agentic split: {split}")
    return AgenticDataset(
        load_dataset(AGENTIC_DATASET, split=split, streaming=True),
        teacher=teacher,
        answer_cache_dir=answer_cache_dir,
        start=start,
        count=count,
    )


class AgenticDataset:
    def __init__(self, samples, *, teacher, answer_cache_dir, start=0, count=None):
        if start < 0 or (count is not None and count < 0):
            raise ValueError("Agentic range must be non-negative")
        self.teacher = teacher
        self.answer_cache_dir = (
            Path(answer_cache_dir).expanduser()
            if answer_cache_dir is not None
            else None
        )
        self.rows = []
        seen = set()
        stop = float("inf") if count is None else start + count
        if count == 0:
            return
        for sample in samples:
            row = self._row(sample)
            if row is None:
                continue
            key = (row["context"], row["question"][0])
            if key in seen:
                continue
            seen.add(key)
            if len(seen) <= start:
                continue
            self.rows.append(row)
            if len(seen) >= stop:
                break

    @staticmethod
    def _row(sample):
        for message in sample.get("prompt", []):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if not isinstance(content, str) or "\n\n" not in content:
                return None
            context, question = content.rsplit("\n\n", 1)
            question = question.strip()
            if not context or not question:
                return None
            return {"context": context, "question": [question], "answers": None}
        return None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    @staticmethod
    def _serializable(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, dict):
            return {
                str(key): AgenticDataset._serializable(item)
                for key, item in value.items()
            }
        if isinstance(value, (tuple, list)):
            return [AgenticDataset._serializable(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return repr(value)

    @staticmethod
    def _first_value(*values):
        return next((str(value) for value in values if value), None)

    @staticmethod
    def _immutable_revision(*values):
        return next(
            (
                value
                for value in values
                if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{40}", value)
            ),
            None,
        )

    def _identity(self, row):
        if self.teacher is None:
            raise ValueError("Agentic answers require a bound runtime teacher")
        from data.wrapper import get_query

        model = getattr(self.teacher, "model", None)
        config = getattr(model, "config", None)
        tokenizer = getattr(self.teacher, "tokenizer", None)
        tokenizer_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
        model_id = self._first_value(
            getattr(model, "_fastkvzip_canonical_id", None),
            getattr(model, "name_or_path", None),
            getattr(model, "model_id", None),
            getattr(self.teacher, "model_id", None),
            getattr(config, "_name_or_path", None),
            getattr(config, "name_or_path", None),
        )
        model_revision = self._immutable_revision(
            getattr(model, "_fastkvzip_revision", None),
            getattr(config, "_commit_hash", None),
            getattr(model, "_commit_hash", None),
            getattr(config, "revision", None),
            getattr(model, "revision", None),
            getattr(self.teacher, "model_revision", None),
            getattr(self.teacher, "revision", None),
        )
        tokenizer_id = self._first_value(
            getattr(tokenizer, "_fastkvzip_canonical_id", None),
            getattr(tokenizer, "name_or_path", None),
            tokenizer_kwargs.get("name_or_path"),
            tokenizer_kwargs.get("_name_or_path"),
        )
        tokenizer_revision = self._immutable_revision(
            getattr(tokenizer, "_fastkvzip_revision", None),
            getattr(tokenizer, "_commit_hash", None),
            tokenizer_kwargs.get("_commit_hash"),
            getattr(tokenizer, "revision", None),
            tokenizer_kwargs.get("revision"),
        )
        if self.answer_cache_dir is not None and not (
            model_id and tokenizer_id and model_revision and tokenizer_revision
        ):
            raise ValueError(
                "persistent Agentic cache requires canonical IDs and immutable model and "
                "tokenizer revisions; use a Hugging Face Hub model ID or omit "
                "--answer-cache-dir for a local model"
            )
        query = get_query("qa", row["question"][0])
        template_ids = self.teacher.apply_template(query)
        content = json.dumps(
            {"context": row["context"], "question": row["question"][0]},
            sort_keys=True,
            separators=(",", ":"),
        )
        return {
            "dataset": AGENTIC_DATASET,
            "content": hashlib.sha256(content.encode()).hexdigest(),
            "model": model_id,
            "model_revision": model_revision,
            "tokenizer": tokenizer_id,
            "tokenizer_revision": tokenizer_revision,
            "prefix": self._serializable(getattr(self.teacher, "sys_prompt_ids", None)),
            "query": query,
            "suffix": self._serializable(getattr(self.teacher, "postfix_ids", None)),
            "template": self._serializable(template_ids),
            "generation_revision": GENERATION_REVISION,
            "generation": self._serializable(getattr(self.teacher, "gen_kwargs", {})),
        }, template_ids

    @staticmethod
    def _cache_key(identity):
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _load_cached_answer(self, identity):
        if self.answer_cache_dir is None:
            return None
        path = self.answer_cache_dir / f"{self._cache_key(identity)}.json"
        if not path.exists():
            return None
        try:
            with path.open() as handle:
                entry = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"corrupt Agentic answer cache: {path}") from error
        if not isinstance(entry, dict) or entry.get("identity") != identity:
            return None
        if not isinstance(entry.get("answer"), str):
            raise ValueError(f"corrupt Agentic answer cache: {path}")
        return entry["answer"]

    def _cache_answer(self, identity, answer):
        if self.answer_cache_dir is None:
            return
        self.answer_cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.answer_cache_dir / f"{self._cache_key(identity)}.json"
        fd, temporary = tempfile.mkstemp(
            dir=self.answer_cache_dir, prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(
                    {"identity": identity, "answer": answer}, handle, sort_keys=True
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def resolve_answers(self, index, full_kv):
        row = self.rows[index]
        if row["answers"] is not None:
            return row["answers"]
        identity, query = self._identity(row)
        answer = self._load_cached_answer(identity)
        if answer is None:
            answer = self.teacher.generate(query, kv=full_kv)
            self._cache_answer(identity, answer)
        answers = [answer]
        if self.answer_cache_dir is not None:
            row["answers"] = answers
        return answers


def _benchmark_range(rows, n_data, start):
    stop = None if n_data is None else start + n_data
    return BenchmarkDataset(rows[start:stop], full_size=len(rows))


def load_squad(n_data=None, *, start=0):
    data = load_dataset("rajpurkar/squad", split="train")

    pool = {}
    for d in data:
        # Group before limiting: later rows may add questions to a selected context.
        row = pool.setdefault(
            d["context"], {"context": d["context"], "question": [], "answers": []}
        )
        row["question"].append(d["question"])
        row["answers"].append(d["answers"]["text"][0])
    return _benchmark_range(list(pool.values()), n_data, start)


def load_gsm(tokenizer, n_data=None, *, start=0):
    dataset_full = load_dataset("openai/gsm8k", "main", split="test")

    dataset = []
    for sample in dataset_full:
        data = dict(sample)
        st = data["question"].split(". ")

        data["context"] = ". ".join(st[:-1]).strip() + "."
        l = len(tokenizer.encode(data["context"], add_special_tokens=False))
        if l < 72:  # pass short context
            continue

        data["question"] = [st[-1].strip()]
        data["answers"] = [data["answer"]]
        dataset.append(data)

    return _benchmark_range(dataset, n_data, start)


def load_scbench(name, n_data=None, *, start=0):
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

    return _benchmark_range(dataset, n_data, start)


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

    samples = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        data_files="sample/10BT/000_00000.parquet",
        split="train",
    )

    length = np.array(samples.data.column("token_count"))
    if "10k" in name:
        min_len, max_len = 10000, 30000
    elif "100k" in name:
        min_len, max_len = 100000, 125000
    else:
        raise AssertionError("check fineweb dataset name!")

    valid = np.arange(len(length))[(length >= min_len) & (length < max_len)]

    total = 0
    dataset = []
    text, token_count = "", 0
    for i in valid.tolist():
        if "cat" in name:
            text += "\n\n" + samples[i]["text"].strip()
            token_count += samples[i]["token_count"]
            if token_count < 100000:
                continue
        else:
            text = samples[i]["text"].strip()
            token_count = samples[i]["token_count"]

        d = {}
        d["context"] = text
        d["question"] = [""]  # only the first question matters now
        d["answers"] = [""]
        dataset.append(d)
        total += token_count

        text, token_count = "", 0
        if total > 10**6:
            break

    return dataset


def load_fineweb_training(train_context_count=29, train_context_start=0):
    if train_context_count < 1:
        raise ValueError("train context count must be positive")
    if train_context_start < 0:
        raise ValueError("train context start must be non-negative")
    samples = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        data_files="sample/10BT/000_00000.parquet",
        split="train",
    )
    lengths = np.array(samples.data.column("token_count"))
    valid = np.flatnonzero((lengths >= 10_000) & (lengths < 30_000)).tolist()
    regular_train = valid[
        train_context_start : train_context_start + train_context_count
    ]
    if len(regular_train) < train_context_count:
        raise ValueError("FineWeb source does not contain enough training contexts")

    def example(text):
        return {"context": text, "question": [""], "answers": [""]}

    def concatenate(start, target=None):
        dataset, text, total, group_tokens, group_start = {}, "", 0, 0, None
        for i in valid:
            if i < start:
                continue
            if group_start is None:
                group_start = i
            text += "\n\n" + samples[i]["text"].strip()
            group_tokens += int(lengths[i])
            if group_tokens < 100_000:
                continue
            dataset[group_start] = example(text)
            total += group_tokens
            if target is None or total >= target:
                return dataset, i
            text, group_tokens, group_start = "", 0, None
        raise ValueError("FineWeb source does not contain enough contexts")

    concat_train, concat_last = concatenate(
        regular_train[0], sum(lengths[regular_train])
    )
    validation_start = max(regular_train[-1], concat_last) + 1
    regular_validation = [i for i in valid if i >= validation_start][:3]
    if len(regular_validation) < 3:
        raise ValueError("FineWeb source does not contain enough validation contexts")
    concat_validation, _ = concatenate(validation_start)

    regular = {
        i: example(samples[i]["text"].strip())
        for i in regular_train + regular_validation
    }
    datasets = {
        "fineweb_10k": regular,
        "fineweb_10k_cat": {**concat_train, **concat_validation},
    }
    train_keys = tuple(
        [("fineweb_10k", i) for i in regular_train]
        + [("fineweb_10k_cat", i) for i in concat_train]
    )
    validation_keys = tuple(
        [("fineweb_10k", i) for i in regular_validation]
        + [("fineweb_10k_cat", next(iter(concat_validation)))]
    )
    return datasets, train_keys, validation_keys


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


def load_mrcr(tokenizer, n_data=2400, max_tokens=128000, n_needles=None, *, start=0):
    """Load MRCR dataset filtered by actual token count"""
    if start < 0 or (n_data is not None and n_data < 0):
        raise ValueError("MRCR range must be non-negative")
    if n_data == 0:
        return BenchmarkDataset([], full_size=None)
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

        if n_data is not None and len(data_list) >= start + n_data:
            # A bounded read has not established the complete filtered size.
            full_size = None
            break
    else:
        full_size = len(data_list)

    return BenchmarkDataset(data_list[start:], full_size=full_size)


if __name__ == "__main__":
    import argparse

    from data.benchmarks import get_data_list
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
