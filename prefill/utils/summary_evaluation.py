"""Shared sampled-answer evaluation, independent of the cache pruning method."""

import hashlib
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm

from data.benchmarks import SUMMARY_DATASETS, get_data_list
from generation import GenerationSettings, SAMPLING_REVISION
from results.evaluation_run import atomic_write_json
from results.sample_store import SampleStore
from utils.tester import Evaluator


def validate_generation_args(args):
    """Validate before loading checkpoints, weights, or datasets."""
    settings = GenerationSettings.from_args(args)
    names = get_data_list(args.data)
    if settings.num_generations > 1 and any(name not in SUMMARY_DATASETS for name in names):
        raise ValueError("num_generations > 1 requires summarization datasets")
    if any(name in SUMMARY_DATASETS for name in names) and not getattr(args, "run_dir", None):
        raise ValueError("summarization evaluation requires --run-dir for sample checkpoints")
    ratios = list(dict.fromkeys(getattr(args, "ratios", None) or []))
    if any(not 0 < ratio <= 1 for ratio in ratios):
        raise ValueError("retention ratios must be in (0, 1]")
    return settings


def configure_generation(model, args):
    """Keep existing benchmark output limits unless explicitly overridden."""
    settings = GenerationSettings.from_args(args)
    if not generation_manifest(args):
        return
    kwargs = settings.hf_kwargs()
    if getattr(args, "max_new_tokens", None) is None and args.data not in SUMMARY_DATASETS:
        kwargs.pop("max_new_tokens")
    model.gen_kwargs.update(kwargs)


def generation_manifest(args):
    """Bind changed decoding to durable run identity without changing legacy runs."""
    settings = GenerationSettings.from_args(args)
    if (any(name in SUMMARY_DATASETS for name in get_data_list(args.data))
            or settings.temperature != 0 or settings.top_p != 1 or settings.top_k != 0
            or getattr(args, "max_new_tokens", None) is not None):
        return {"generation_settings": settings.as_dict()}
    return {}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _file_digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _tokens(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return value


def _runtime_identity(model):
    runtime = model.model
    config = runtime.config
    tokenizer = model.tokenizer
    tokenizer_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
    identity = {
        "model": getattr(runtime, "_fastkvzip_canonical_id", None) or getattr(runtime, "name_or_path", None),
        "model_revision": getattr(runtime, "_fastkvzip_revision", None) or getattr(config, "_commit_hash", None),
        "tokenizer": getattr(tokenizer, "_fastkvzip_canonical_id", None) or getattr(tokenizer, "name_or_path", None),
        "tokenizer_revision": getattr(tokenizer, "_fastkvzip_revision", None) or tokenizer_kwargs.get("_commit_hash"),
        "chat_template": getattr(tokenizer, "chat_template", None),
        "prefix": _tokens(model.sys_prompt_ids),
        "postfix": _tokens(model.postfix_ids),
        "dtype": str(getattr(model, "dtype", None)),
        "sampling_revision": SAMPLING_REVISION,
        "model_config": config.to_dict() if hasattr(config, "to_dict") else vars(config).copy(),
        "tokenizer_cleanup": getattr(tokenizer, "clean_up_tokenization_spaces", None),
    }
    if (not identity["model"] or not identity["tokenizer"] or any(
            not isinstance(identity[key], str) or not re.fullmatch(r"[0-9a-fA-F]{40}", identity[key])
            for key in ("model_revision", "tokenizer_revision"))):
        raise ValueError("resumable summaries require model and tokenizer IDs with immutable revisions")
    model_generation_config = getattr(runtime, "generation_config", None)
    generation_config = model.gen_kwargs.get("generation_config") or model_generation_config
    eos = getattr(generation_config, "eos_token_id", None)
    if eos is None:
        eos = getattr(model_generation_config, "eos_token_id", None)
    if eos is None:
        eos = getattr(config, "eos_token_id", None)
    identity["eos_token_id"] = model.gen_kwargs.get("eos_token_id", eos)
    return json.loads(json.dumps(identity, allow_nan=False))


def _pruning_identity(run, model, args):
    identity = dict(run.manifest)
    checkpoint = identity.get("checkpoint_path")
    if checkpoint:
        identity["checkpoint_sha256"] = _file_digest(checkpoint)
    for name in ("prefill_chunk", "token_microbatch_size", "graph_microbatch_size"):
        identity[name] = getattr(args, name, None)
    gate = getattr(args, "gate_path_or_name", None)
    if gate and Path(gate).is_file():
        identity["gate_sha256"] = _file_digest(gate)
    elif getattr(model, "gates", None):
        fingerprint = hashlib.sha256()
        for gate_module in model.gates:
            for name, tensor in gate_module.state_dict().items():
                fingerprint.update(name.encode())
                fingerprint.update(str((tensor.dtype, tuple(tensor.shape))).encode())
                fingerprint.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        identity["gate_sha256"] = fingerprint.hexdigest()
    return identity


def _context_limit(model):
    config = model.config
    text_config = getattr(config, "text_config", config)
    candidates = [getattr(text_config, "max_position_embeddings", None),
                  getattr(model.tokenizer, "model_max_length", None)]
    limits = [int(value) for value in candidates
              if isinstance(value, (int, float)) and 0 < value < 10**9]
    if not limits:
        raise ValueError("cannot establish the model context limit for complete-document evaluation")
    return min(limits)


def evaluate_summary_dataset(dataset, args, run, cache_provider, *, evaluator_factory=Evaluator):
    """Evaluate a dataset using a provider yielding (ratio, cache, actual retention).

    Providers prepare only the requested missing conditions and may reuse a
    scored full cache across ratios. They never depend on the sample count.
    """
    settings = GenerationSettings.from_args(args)
    task = dataset.name
    model = dataset.model
    runtime = _runtime_identity(model)
    protocol = {"runtime": runtime, "generation": settings.as_dict(),
                "pruning": _pruning_identity(run, model, args)}
    directory = run.run_dir / "samples" / task
    manifest_path = directory / "manifest.json"
    context_limit = _context_limit(model)
    eligible, excluded, inventory, prepared = [], [], [], {}
    for index, row in enumerate(dataset.dataset):
        inputs, info = dataset.generate_answer(index, None, prob=False, full_cache_answer=False)
        task_query = next(iter(info))
        query = inputs[task_query]["q"]
        context_ids = model.tokenizer.encode(row["context"], add_special_tokens=False)
        prompt_ids = runtime["prefix"] + _tokens(context_ids) + _tokens(query)
        metadata = {"index": index, "document_id": row["id"],
                    "document_tokens": len(context_ids), "prompt_tokens": len(prompt_ids),
                    "length_bin": row["length_bin"]}
        inventory.append(metadata)
        if len(prompt_ids) + settings.max_new_tokens > context_limit:
            excluded.append({**metadata, "reason": "context_limit"})
            continue
        eligible.append(index)
        identity = {**protocol, "input_sha256": _digest(prompt_ids),
                    "document_sha256": hashlib.sha256(row["context"].encode()).hexdigest(),
                    "source": row.get("source")}
        prepared[index] = (metadata, identity)
    stop = None if args.num is None else args.idx + args.num
    selected = eligible[args.idx:stop]
    ratios = list(dict.fromkeys([1.0, *args.ratios]))
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["protocol"] != protocol or previous["inventory"] != inventory:
            raise ValueError("summary protocol or dataset inventory changed; use a new run directory")
        selected = sorted(set(previous["selected_indices"]) | set(selected))
        ratios = list(dict.fromkeys([*previous["requested_ratios"], *ratios]))
    manifest = {"protocol": protocol, "num_generations": settings.num_generations,
                "requested_ratios": ratios, "selected_indices": selected,
                "dataset_size": len(eligible), "excluded": excluded, "inventory": inventory,
                "source_inventory": getattr(dataset.dataset, "inventory", None),
                "example_identities": {str(i): _digest(prepared[i][1]) for i in selected},
                "context_limit": context_limit}
    # Validate every existing sample identity before starting further GPU work.
    stores = {index: SampleStore(directory / "examples" / f"{index}.json",
                                identity=prepared[index][1], num_generations=settings.num_generations,
                                metadata=prepared[index][0]) for index in selected}
    atomic_write_json(manifest_path, manifest)
    run.record_dataset_size(task, len(eligible))
    print(f"{task}: {len(eligible)} eligible, {len(excluded)} over model limit, {len(selected)} selected")
    source_inventory = manifest["source_inventory"]
    if source_inventory:
        counts = {key: source_inventory[key] for key in ("source_size", "eligible_size", "excluded", "length_bins")}
        print(f"Source document inventory: {counts}")
    for index in tqdm(selected, desc=task):
        store = stores[index]
        full_cache_store = None
        cache_dir = getattr(args, "answer_cache_dir", None)
        if cache_dir:
            full_identity = {key: value for key, value in prepared[index][1].items() if key != "pruning"}
            full_cache_store = SampleStore(Path(cache_dir) / "summaries" / f"{_digest(full_identity)}.json",
                                          identity=full_identity, num_generations=settings.num_generations,
                                          metadata={})
            for sample in full_cache_store.data["ratios"].get("1.0", {}).get("samples", []):
                store.add_sample(1, sample, actual_retention=1)
            for sample in store.data["ratios"].get("1.0", {}).get("samples", []):
                full_cache_store.add_sample(1, sample, actual_retention=1)
        missing = [ratio for ratio in ratios if not store.is_complete(ratio)]
        if not missing:
            continue
        inputs, info = dataset.generate_answer(index, None, prob=False, full_cache_answer=False)
        evaluator = evaluator_factory(model, inputs, info)
        task_query = next(iter(info))
        provider = iter(cache_provider(index, missing))
        try:
            for ratio, kv, actual_retention in provider:
                if ratio not in missing or store.is_complete(ratio):
                    raise ValueError("cache provider returned an unexpected or complete ratio")
                # Full-reference seeds are independent of the pruner and its budget.
                seed_identity = {key: value for key, value in prepared[index][1].items()
                                 if ratio != 1 or key != "pruning"}
                seed = int(_digest({"identity": seed_identity, "ratio": ratio})[:15], 16)

                def save(sample):
                    store.add_sample(ratio, sample, actual_retention=actual_retention)
                    if ratio == 1 and full_cache_store is not None:
                        full_cache_store.add_sample(1, sample, actual_retention=1)

                evaluator.generation_samples(kv, task_query, settings,
                                             sample_indices=store.missing_indices(ratio),
                                             seed=seed, on_sample=save)
                if not store.is_complete(ratio):
                    raise ValueError(f"generation returned an incomplete pool for ratio {ratio}")
                del kv  # Release a chunked cache before the provider allocates the next one.
        finally:
            if hasattr(provider, "close"):
                provider.close()
        if any(not store.is_complete(ratio) for ratio in ratios):
            raise ValueError("cache provider did not produce every requested ratio")
    return len(eligible)
