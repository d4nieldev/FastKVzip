from collections import defaultdict
from typing import List, Optional, Tuple, Union

import torch
from attention.kvcache import EvictCache, RetainCache
from model import ModelKVzip


def get_query(task, q=None):
    if task == "repeat":
        query = f"Repeat the previous context exactly."
    elif task == "qa":
        if q is None:
            query = f"Q: Answer the question based on the previous context."
        else:
            query = f"Q: {q}"
    elif task == "reason":
        query = f"Reason and answer the question. You must say the answer in the last sentence beginning with 'The answer is'. Q: {q}"
    elif task == "summarize":
        query = f"Please summarize the previous context."
    else:
        raise ValueError(f"Invalid task: {task}")

    return query


class DataWrapper:

    def __init__(
        self, dataname, dataset, model: ModelKVzip, *, ruler_prompt_mode="graphkv"
    ):
        if ruler_prompt_mode not in {"graphkv", "official"}:
            raise ValueError("ruler prompt mode must be graphkv or official")
        self.name, self.dataset, self.model = dataname, dataset, model
        self._longbench = dataname.startswith("longbench_")
        self._official_ruler = (
            dataname.startswith("ruler_") and ruler_prompt_mode == "official"
        )
        model.set_chat_template("ruler_official" if self._official_ruler else dataname)
        if dataname == "longbench_v2":
            self._v2_prefix_length = model.sys_prompt_ids.shape[1]
            self._v2_tail_lengths = {}

    def __len__(self):
        return len(self.dataset)

    def prefill_context(
        self,
        idx: int,
        do_score=False,
        prefill_chunk=16000,
        window_size=512,
        chunk_ratio=1.0,
        level="pair",
        save_hidden=False,
        chunk_scorer=None,
    ) -> Union[RetainCache, EvictCache]:
        """Prefill and scoring KV importance"""
        data = self.dataset[idx]
        ctx_ids = self.model.encode(data["context"])

        # The graph evaluator restores its saved prefix after wrapper creation.
        # Add task instructions here so only raw context is scored and evicted.
        prefix_ids = self.model.sys_prompt_ids
        if self._longbench:
            self.model.sys_prompt_ids = torch.cat(
                [prefix_ids, self.model.encode(data["context_prefix"])], dim=1
            )
        try:
            if self.name == "longbench_v2":
                from data.longbench_v2 import MAX_INPUT_TOKENS, MAX_NEW_TOKENS

                capacity = getattr(self.model.config, "max_position_embeddings", None)
                if type(capacity) is not int or capacity <= 0:
                    raise ValueError("LongBench v2 requires a known positive model capacity")
                if idx not in self._v2_tail_lengths:
                    query_length = self.model.encode(data["question"][0]).shape[1]
                    answer_tail = query_length + self.model.postfix_ids.shape[1] + MAX_NEW_TOKENS
                    # KVzip replays 2K-token chunks atop the full cache. Two chunks
                    # cover both its initial and continuation instructions. Reserve
                    # this space for every method, including the full-cache baseline.
                    replay_tail = max(
                        repeat_ids.shape[1]
                        for _, repeat_ids in self.model.self_task(ctx_ids[:, :4000])
                    )
                    self._v2_tail_lengths[idx] = max(answer_tail, replay_tail)
                capacity = min(MAX_INPUT_TOKENS, capacity)
                tail = self._v2_tail_lengths[idx]
                # Use the native prefix budget for identical context across
                # methods, even when GraphKV restores a different saved prefix.
                task_prefix_length = self.model.sys_prompt_ids.shape[1] - prefix_ids.shape[1]
                budget = capacity - self._v2_prefix_length - task_prefix_length - tail
                if budget < 1:
                    raise ValueError("LongBench v2 protected prompt leaves no room for context")
                if ctx_ids.shape[1] > budget:
                    print(f"LongBench v2 context truncated: {ctx_ids.shape[1]} -> {budget} tokens")
                    # Keep an odd leftover token at the front; avoid the -0 slice.
                    ctx_ids = torch.cat(
                        [ctx_ids[:, :(budget + 1) // 2], ctx_ids[:, ctx_ids.shape[1] - budget // 2:]], dim=1
                    )
                if self.model.sys_prompt_ids.shape[1] + ctx_ids.shape[1] + tail > capacity:
                    raise ValueError("LongBench v2 checkpoint prefix exceeds the shared input budget")
            kv = self.model.prefill(
                ctx_ids,
                do_score=do_score,
                prefill_chunk_size=prefill_chunk,
                window_size=window_size,
                chunk_ratio=chunk_ratio,
                level=level,
                save_hidden=save_hidden,
                chunk_scorer=chunk_scorer,
            )
        finally:
            self.model.sys_prompt_ids = prefix_ids

        print(
            f"# prefill {self.model.name} {self.name}-{idx}: "
            f"{len(ctx_ids[0])} tokens, KV cache {kv._mem()} GB, {kv.key_cache[0].dtype}"
        )
        return kv

    def _prepare_query(
        self,
        data,
        kv,
        inputs: dict,
        task: str,
        full_cache_answer: bool = True,
        resolved_full_answers=None,
    ):
        """Generate answers of each task for evaluation.
        For each task, we store (query, answer, grount_truth) in inputs
        """
        if task in ["qa", "reason"]:
            if full_cache_answer:
                print("# Generated output | Ground truth")
            answers = (
                resolved_full_answers
                if resolved_full_answers is not None
                else data["answers"]
            )
            for i, (q, gt) in enumerate(zip(data["question"], answers)):
                if self._longbench:
                    q_ids = torch.cat(
                        [self.model.encode(q), self.model.postfix_ids], dim=1
                    )
                else:
                    q = q if self._official_ruler else get_query(task, q)
                    q_ids = self.model.apply_template(q)

                if full_cache_answer:
                    a = (
                        resolved_full_answers[i]
                        if resolved_full_answers is not None
                        else self.model.generate(q_ids, kv=kv)
                    )
                    a_ids = self.model.encode(a)
                else:
                    a_ids = None
                gt_ids = gt if isinstance(gt, list) else self.model.encode(gt)

                tag = f"qa-{i}" if i > 0 else "qa"
                inputs[tag] = {"q": q_ids, "a": a_ids, "gt": gt_ids}
                inputs["eval_task"].append(tag)

                if full_cache_answer:
                    print(f"[QA {i}] {a} | {gt}")

        else:
            q = get_query(task)
            q_ids = self.model.apply_template(q)

            if task == "repeat":
                a_ids = kv.ctx_ids
            else:
                a = self.model.generate(q_ids, kv=kv)
                a_ids = self.model.encode(a)

            gt_ids = a_ids  # no ground truth
            inputs[task] = {"q": q_ids, "a": a_ids, "gt": gt_ids}
            if "scbench" not in self.name and a_ids.shape[-1] < 512:
                inputs["eval_task"].append(task)

    @torch.inference_mode()
    def generate_answer(
        self,
        idx: int,
        kv: Union[RetainCache, EvictCache],
        prob=True,
        full_cache_answer: bool = True,
    ):
        """Prepare inputs, answers, and prediction probabilities (with full KV cache) for evaluation."""
        if prob and not full_cache_answer:
            raise ValueError("full-cache probabilities require a full-cache answer")
        data = self.dataset[idx]
        resolved_full_answers = None
        if data["answers"] is None:
            resolved_full_answers = self.dataset.resolve_answers(idx, kv)
            data = self.dataset[idx]

        eval_task = ["qa"]
        if "gsm" in self.name:
            eval_task = ["reason"]

        inputs = defaultdict(list)
        for task in eval_task:
            self._prepare_query(
                data,
                kv,
                inputs,
                task,
                full_cache_answer=full_cache_answer,
                resolved_full_answers=resolved_full_answers,
            )

        info = defaultdict(dict)
        for fmt in inputs["eval_task"]:
            info[fmt] = {}
            if prob:
                input_ids = torch.cat([inputs[fmt][k] for k in ["q", "a"]], dim=1)
                info[fmt]["prob"] = self.model._prob(input_ids, kv, device="cpu")

        return inputs, info
