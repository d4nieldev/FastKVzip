# ------------------------------------------------------------------------------
# Original Code developed by Jang-Hyun Kim
# GitHub Repository: https://github.com/snu-mllab/KVzip
# ------------------------------------------------------------------------------
import glob
import copy
import hashlib
from typing import List, Optional, Tuple, Union

import torch
from attention.gate import load_gate
from attention.kvcache import EvictCache, RetainCache, RetainHybridCache
from model.load import load_model
from model.template import template
from tqdm import tqdm
from transformers import (
    DynamicCache,
    Gemma3ForCausalLM,
    Qwen3ForCausalLM,
    Qwen3MoeForCausalLM,
)

from utils.func import inplace_softmax
from window import resolve_window_size
from generation import GenerationSettings


class _PromptCacheState:
    """Rollback append-only caches, copying only metadata and mutable hybrid regions."""

    def __init__(self, kv, incoming_tokens):
        self.keys = list(kv.key_cache)
        self.values = list(kv.value_cache)
        self.metadata = {
            name: copy.deepcopy(getattr(kv, name))
            for name in ('_seen_tokens', '_cur_tokens', 'info', 'cu_len_q')
            if hasattr(kv, name)
        }
        self.mutable = []
        self.prefix_lengths = None
        if isinstance(kv, RetainHybridCache):
            start = kv._seen_tokens
            end = start + incoming_tokens
            for layer, (key, value) in enumerate(zip(self.keys, self.values)):
                region = slice(start, end) if layer in kv.static_layer_ids else slice(None)
                self.mutable.append((layer, region, key[:, :, region].clone(), value[:, :, region].clone()))

    def rebase_on_prompt(self, kv):
        """Drop the original allocation once its prefix lives in the prompt cache."""
        if isinstance(kv, RetainHybridCache):
            return
        self.prefix_lengths = [key.shape[-2] for key in self.keys]
        self.prompt_offsets = (
            copy.deepcopy(kv.info['cu_len_k']) if isinstance(kv, EvictCache) else None
        )
        self.keys = list(kv.key_cache[:len(self.prefix_lengths)])
        self.values = list(kv.value_cache[:len(self.prefix_lengths)])

    def restore(self, kv):
        for layer, region, key, value in self.mutable:
            self.keys[layer][:, :, region].copy_(key)
            self.values[layer][:, :, region].copy_(value)
        if self.prefix_lengths is None:
            kv.key_cache[:] = self.keys
            kv.value_cache[:] = self.values
        else:
            # Release the working decode cache before rebuilding context prefixes.
            kv.key_cache.clear()
            kv.value_cache.clear()
            for layer, length in enumerate(self.prefix_lengths):
                for target, source in ((kv.key_cache, self.keys), (kv.value_cache, self.values)):
                    if self.prompt_offsets is None:
                        restored = source[layer][..., :length, :]
                    else:
                        offsets = self.prompt_offsets[layer]
                        lengths = self.metadata['info']['len_k'][layer]
                        restored = torch.cat([
                            source[layer][offsets[head]:offsets[head] + lengths[head]]
                            for head in range(len(lengths))
                        ])
                    target.append(restored)
        for name, value in self.metadata.items():
            setattr(kv, name, copy.deepcopy(value))


def chunk_fn(ctx_ids: torch.Tensor, chunk_size: int) -> List[torch.Tensor]:
    """Chunk tokens"""
    ctx_len = ctx_ids.shape[1]
    if ctx_len > chunk_size:
        chunk_num = (ctx_len - 1) // chunk_size + 1
        print(f"chunk inputs, size: {chunk_size} (num {chunk_num})")

        input_ids = []
        for i in range(chunk_num):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            a_ids = ctx_ids[:, start:end]
            if a_ids.shape[1] == 0:
                continue
            input_ids.append(a_ids)
    else:
        input_ids = [ctx_ids]

    return input_ids


class ModelKVzip:

    def __init__(
        self, model_name: str, kv_type: str = "evict", gate_path_or_name="fastkvzip"
    ):
        self.model, self.tokenizer = load_model(model_name)

        self.name = self.model.name
        self.dtype = self.model.dtype
        self.device = self.model.device
        self.config = self.model.config

        self.gates = load_gate(self, gate_path_or_name)

        if isinstance(self.model, Gemma3ForCausalLM):
            self.kv_type = "hybrid_static"
            print("[Note] Currently, only retain cache is available for Gemma3")
        else:
            self.kv_type = kv_type
        print(f"KV type: {self.kv_type}")

        self.gen_kwargs = {
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1,
            "top_k": None,
            "max_new_tokens": 512,
        }
        if isinstance(self.model, Gemma3ForCausalLM):
            self.gen_kwargs["cache_implementation"] = None
            self.gen_kwargs["use_model_defaults"] = False
            self.gen_kwargs["eos_token_id"] = [1, 106]
        elif isinstance(self.model, Qwen3ForCausalLM) or isinstance(
            self.model, Qwen3MoeForCausalLM
        ):
            self.gen_kwargs["cache_implementation"] = None
            self.gen_kwargs["use_model_defaults"] = False
            self.gen_kwargs["eos_token_id"] = 151645

        self.set_chat_template()

    def encode(self, text: str) -> torch.Tensor:
        """Encode text into tokens"""
        return self.tokenizer.encode(
            text, add_special_tokens=False, return_tensors="pt"
        ).cuda()

    def decode(self, input_ids: torch.Tensor) -> str:
        """Decode tokens into text"""
        if len(input_ids.shape) == 2:
            input_ids = input_ids[0]
        return self.tokenizer.decode(input_ids)

    def set_chat_template(self, task: str = "qa"):
        prefix, postfix = template(self.name, task)
        self.sys_prompt_ids, self.postfix_ids = self.encode(prefix), self.encode(
            postfix
        )

    def apply_template(self, query: str) -> torch.Tensor:
        query = f"\n\n{query.strip()}"
        query_ids = torch.cat([self.encode(query), self.postfix_ids], dim=1)
        return query_ids

    def __call__(
        self,
        input_ids: torch.Tensor,
        kv: Union[RetainCache, EvictCache],
        update_cache: bool = False,
        return_logits: bool = False,
        *args,
        **kwargs,
    ):
        """Compute Transformer forward pass
        In default, we do not update the KV cache with the newly given inputs.
        Set update_cache = True to enable the update.
        """
        seen_token_prev = kv._seen_tokens

        if isinstance(kv, RetainHybridCache) and not update_cache:
            kv.backup_sliding_cache()

        if return_logits:
            outputs = self.model(input_ids, past_key_values=kv, *args, **kwargs)
        else:
            _ = self.model.model(input_ids, past_key_values=kv, *args, **kwargs)
            outputs = None

        if not update_cache:
            kv.slice(seen_token_prev)
        return outputs

    def _init_kv(self, kv=None, evict_range=(0, 0)):
        """Initialize KV cache"""

        if kv is None:
            if self.kv_type == "retain":
                kv = RetainCache(self.model, evict_range)
            elif self.kv_type == "evict":
                kv = EvictCache(self.model, evict_range)
            elif self.kv_type == "hybrid_static":
                max_size = 190000
                kv = RetainHybridCache(self.model.model, evict_range, max_size)
            elif self.kv_type == "original":
                kv = DynamicCache()
                kv.pruned, kv.get_score = False, False
            else:
                raise NotImplementedError(f"type {self.kv_type} is not implemented")
        return kv

    @torch.inference_mode()
    def prefill(
        self,
        ctx_ids: Union[str, torch.Tensor],
        prefill_chunk_size: int = 16000,
        do_score=False,
        window_size=4096,
        window_ratio=0.02,
        chunk_ratio=1.0,
        level="pair",
        save_hidden=False,
        chunk_scorer=None,
    ) -> Union[RetainCache, EvictCache]:
        """Chunked prefill KV cache"""
        if type(ctx_ids) == str:
            ctx_ids = self.encode(ctx_ids)
        prefill_ids = torch.cat([self.sys_prompt_ids, ctx_ids], dim=1)
        evict_range = (self.sys_prompt_ids.shape[1], prefill_ids.shape[1])

        kv = self._init_kv(evict_range=evict_range)  # do not evict system prompt KV
        kv.ctx_ids = ctx_ids
        kv.prefill_ids = prefill_ids

        ########### Chunked scoring + evict ###########
        kv.save_hidden = save_hidden or chunk_scorer is not None
        kv.gates = self.gates
        if self.gates is not None or chunk_scorer is not None:
            kv.init_score(get_score=False)
            start_idx = evict_range[0]
            clen = kv.ctx_len
            window_size = resolve_window_size(
                window_size, clen, prefill_chunk_size, window_ratio
            )

        if chunk_ratio < 1.0:
            # adjust compression ratio considering a local window
            if chunk_ratio * clen < window_size:
                window_size = int(chunk_ratio * clen)
                chunk_ratio = 0.0
            else:
                chunk_ratio = (chunk_ratio * clen - window_size) / (clen - window_size)

        # prefill
        for input_ids in tqdm(
            chunk_fn(prefill_ids, prefill_chunk_size), desc="Prefill"
        ):
            self.__call__(input_ids, kv, update_cache=True)
            if chunk_scorer is not None:
                chunk_scorer(kv)

            if chunk_ratio < 1.0:
                end_idx = max(start_idx, kv.score[0].shape[-1] - window_size)
                if end_idx > start_idx:
                    kv.prune_chunk(chunk_ratio, (start_idx, end_idx), level)
                    start_idx = end_idx

        if chunk_ratio < 1.0 and self.kv_type != "evict":
            valid = kv.valid.new_ones((*kv.valid.shape[:-1], window_size))
            kv.valid = torch.cat([kv.valid, valid], dim=-1)
            assert kv.valid.size(-1) == kv.ctx_len
            ratio = kv.valid.float().mean()
            print(f"Chunked prefilling with {ratio:.2f} ratio, window {window_size}")

        if chunk_scorer is not None:
            kv.hidden_cache.clear()
        kv.save_hidden, kv.compute_gate = False, False
        #######################################

        if do_score:
            if self.gates is None:
                self.scoring(kv, ctx_ids)
            else:
                kv.score = torch.stack(kv.score, dim=0)
                kv.score = kv.score[..., start_idx:]
                window_size = max(0, min(window_size, kv.ctx_len))
                if window_size > 0:
                    kv.score[..., -window_size:] = kv.score.max()
                kv.protected_window = window_size
                print(f"Local window {window_size}")

        return kv

    def self_task(
        self,
        ctx_ids: torch.Tensor,
        chunk_size: int = 2000,
        prev_postfix_size=8,
    ) -> List[torch.Tensor]:
        """Prepare chunked inputs for KV importance scoring with context reconstruction
        return: List[torch.Tensor]
        """
        chunked_inputs = chunk_fn(ctx_ids, chunk_size)

        input_ids = []
        for i, a_ids in enumerate(chunked_inputs):
            if i == 0:
                prompt = f"\n\nRepeat the previous context exactly."
                q_ids = self.encode(prompt)
            else:
                prompt = f"\n\nRepeat the part of the previous context exactly, starting with "
                q_ids = self.encode(prompt)
                postfix_prev = chunked_inputs[i - 1][:, -prev_postfix_size:]
                q_ids = torch.cat([q_ids, postfix_prev], dim=1)

            input_ids.append(
                (a_ids, torch.cat([q_ids, self.postfix_ids, a_ids], dim=1))
            )

        return input_ids

    @torch.inference_mode()
    def scoring(self, kv: Union[RetainCache, EvictCache], ctx_ids: torch.Tensor):
        """KVzip importance scoring (update kv.score)"""
        kv.init_score()
        start_idx_tmp = kv.start_idx

        kv.end_idx = 0
        input_ids = self.self_task(ctx_ids)
        for i, (prefill_ids_p, repeat_ids_p) in enumerate(
            tqdm(input_ids, desc=f"Importance scoring")
        ):
            kv.end_idx = kv.start_idx + prefill_ids_p.shape[1]  # indices for a chunk
            self.__call__(repeat_ids_p, kv, update_cache=False)  # get score
            kv.start_idx = kv.end_idx

        kv.start_idx = start_idx_tmp
        assert kv.score[0].shape[-1] == kv.ctx_len
        kv.get_score = False

    @torch.inference_mode()
    def generate(
        self,
        query: Union[str, torch.Tensor],
        kv: Optional[Union[RetainCache, EvictCache]] = None,
        update_cache: bool = False,
    ) -> str:
        """Obtain a model response to the query
        In default, we evict KV of query and generated answer after the generation by kv.slice (for multi-query evaluation).
        Set update_cache = True to enable multi-turn generation.
        """
        kv = self._init_kv(kv=kv)
        seen_token_prev = kv._seen_tokens

        if isinstance(kv, RetainHybridCache) and not update_cache:
            kv.backup_sliding_cache()

        input_ids = query
        if type(query) == str:
            input_ids = self.encode(query)
        if kv.prefill_ids is not None:
            # Huggingface Transformers model.generate requires full input tokens when using KV caches.
            # The inputs will be spliced to only contain new tokens as input[:, -kv.get_seq_length():].
            input_ids = torch.cat([kv.prefill_ids, input_ids], dim=1)

        output = self.model.generate(input_ids, past_key_values=kv, **self.gen_kwargs)
        a_ids = output[:, len(input_ids[0]) :]
        model_generation_config = getattr(self.model, "generation_config", None)
        generation_config = (
            self.gen_kwargs.get("generation_config") or model_generation_config
        )
        eos_ids = getattr(generation_config, "eos_token_id", None)
        if eos_ids is None:
            eos_ids = getattr(model_generation_config, "eos_token_id", None)
        eos_ids = self.gen_kwargs.get("eos_token_id", eos_ids)
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        if a_ids.size(1) and eos_ids is not None and a_ids[0, -1].item() in eos_ids:
            a_ids = a_ids[:, :-1]
        a = self.decode(a_ids)

        if not update_cache:
            kv.slice(seen_token_prev)
        else:
            kv.prefill_ids = torch.cat([input_ids, a_ids], dim=1)
        return a

    @torch.inference_mode()
    def sample_responses(
        self, query, kv, settings, *, sample_indices=None, seed=0, on_sample=None
    ):
        """Prefill a query once, then sequentially sample reproducible continuations.

        Token counts include a generated EOS; decoded text excludes the final EOS.
        The supplied context cache is restored even if a forward or callback fails.
        """
        from transformers import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

        if not isinstance(settings, GenerationSettings):
            raise TypeError('settings must be GenerationSettings')
        indices = list(range(settings.num_generations) if sample_indices is None else sample_indices)
        if any(type(index) is not int or not 0 <= index < settings.num_generations for index in indices):
            raise ValueError('sample_indices must be integers in [0, num_generations)')
        if len(set(indices)) != len(indices):
            raise ValueError('sample_indices must be unique')
        if type(seed) is not int:
            raise ValueError('seed must be an integer')
        if not indices:
            return []
        input_ids = self.encode(query) if isinstance(query, str) else query
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError('query must contain one nonempty token sequence')

        model_config = getattr(self.model, 'generation_config', None)
        generation_config = self.gen_kwargs.get('generation_config') or model_config
        eos_ids = getattr(generation_config, 'eos_token_id', None)
        if eos_ids is None:
            eos_ids = getattr(model_config, 'eos_token_id', None)
        if eos_ids is None:
            eos_ids = getattr(self.model.config, 'eos_token_id', None)
        eos_ids = self.gen_kwargs.get('eos_token_id', eos_ids)
        eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
        warpers = []
        if settings.do_sample:
            if settings.temperature != 1:
                warpers.append(TemperatureLogitsWarper(float(settings.temperature)))
            if settings.top_k:
                warpers.append(TopKLogitsWarper(settings.top_k))
            if settings.top_p < 1:
                warpers.append(TopPLogitsWarper(settings.top_p))

        def forward(tokens):
            positions = torch.arange(kv._seen_tokens, kv._seen_tokens + tokens.shape[1], device=tokens.device)
            return self(tokens, kv, update_cache=True, return_logits=True,
                        use_cache=True, cache_position=positions).logits[:, -1, :].clone()

        context = _PromptCacheState(kv, input_ids.shape[1] + settings.max_new_tokens)
        samples = []
        try:
            # This forward sees the active pruning mask, including for token one.
            initial_logits = forward(input_ids)
            context.rebase_on_prompt(kv)
            prompt = _PromptCacheState(kv, settings.max_new_tokens)
            for index in indices:
                prompt.restore(kv)
                sample_seed = int.from_bytes(hashlib.sha256(f'fastkvzip:{seed}:{index}'.encode()).digest()[:8], 'big') % (2**63)
                device = initial_logits.device
                rng_device = device if device.type in ('cpu', 'cuda') else torch.device('cpu')
                generator = torch.Generator(device=rng_device).manual_seed(sample_seed)
                logits = initial_logits
                token_ids = []
                finish_reason = 'length'
                for step in range(settings.max_new_tokens):
                    if settings.do_sample:
                        scores = logits.float()
                        for warper in warpers:
                            scores = warper(input_ids, scores)
                        probabilities = torch.softmax(scores, dim=-1).to(rng_device)
                        token = torch.multinomial(probabilities, 1, generator=generator).to(device)
                    else:
                        token = logits.argmax(dim=-1, keepdim=True)
                    token_id = token.item()
                    token_ids.append(token_id)
                    if token_id in eos_ids:
                        finish_reason = 'eos'
                        break
                    if step + 1 < settings.max_new_tokens:
                        logits = forward(token)
                text_ids = token_ids[:-1] if finish_reason == 'eos' else token_ids
                sample = {
                    'index': index, 'seed': sample_seed,
                    'text': self.decode(torch.tensor([text_ids], dtype=torch.long, device=input_ids.device)),
                    'token_count': len(token_ids), 'finish_reason': finish_reason,
                    'token_ids': token_ids,
                }
                samples.append(sample)
                if on_sample is not None:
                    on_sample(sample)
            return samples
        finally:
            context.restore(kv)

    @torch.inference_mode()
    def _prob(self, input_ids, kv=None, device="cuda") -> torch.Tensor:
        """Obtain next token prediction probabilities"""
        kv = self._init_kv(kv=kv)

        output = self.__call__(input_ids, kv, update_cache=False, return_logits=True)
        output = output.logits[0]
        output = inplace_softmax(output).squeeze()

        if device == "cpu":
            return output.cpu()
        return output
