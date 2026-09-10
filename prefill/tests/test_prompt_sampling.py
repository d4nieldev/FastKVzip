import argparse
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generation import GenerationSettings, add_generation_arguments


class SettingsTests(unittest.TestCase):
    def test_defaults_and_cli(self):
        parser = argparse.ArgumentParser()
        add_generation_arguments(parser)
        self.assertEqual(GenerationSettings.from_args(parser.parse_args([])), GenerationSettings())
        settings = GenerationSettings.from_args(parser.parse_args(['--temperature', '.8', '--top-p', '.9', '--top-k', '12', '--num-generations', '3']))
        self.assertTrue(settings.do_sample)
        self.assertEqual(settings.max_new_tokens, 1024)

    def test_invalid_and_greedy(self):
        for kwargs in ({'temperature': -1}, {'temperature': float('nan')}, {'top_p': 0}, {'top_p': 1.1}, {'top_k': -1}, {'top_k': 1.2}, {'max_new_tokens': 0}, {'num_generations': 0}, {'num_generations': 2}, {'temperature': 1, 'top_k': 1, 'num_generations': 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GenerationSettings(**kwargs)
        self.assertFalse(GenerationSettings(temperature=1, top_k=1).do_sample)

import torch
from transformers import LlamaConfig, LlamaForCausalLM
from attention.kvcache import RetainCache, EvictCache, RetainHybridCache
from model.wrapper import ModelKVzip


class SamplingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        config = LlamaConfig(vocab_size=19, hidden_size=16, intermediate_size=24,
                             num_hidden_layers=2, num_attention_heads=2,
                             num_key_value_heads=1, eos_token_id=None)
        config._attn_implementation = 'eager'
        self.wrapper = ModelKVzip.__new__(ModelKVzip)
        self.wrapper.model = LlamaForCausalLM(config).eval()
        self.wrapper.gen_kwargs = {}
        self.wrapper.decode = lambda ids: ' '.join(map(str, ids.flatten().tolist()))
        self.kv = RetainCache(self.wrapper.model, (0, 3))
        self.wrapper(torch.tensor([[2, 3, 4]]), self.kv, update_cache=True)
        self.original = [x.clone() for x in self.kv.key_cache]
        self.query = torch.tensor([[5, 6]])

    def assert_restored(self):
        self.assertEqual(self.kv._seen_tokens, 3)
        for old, new in zip(self.original, self.kv.key_cache):
            torch.testing.assert_close(old, new)

    def test_prompt_once_resume_seeds_and_max(self):
        calls = []
        handle = self.wrapper.model.register_forward_pre_hook(lambda model, args: calls.append(args[0].shape[1]))
        settings = GenerationSettings(temperature=.8, max_new_tokens=4, num_generations=3)
        rng_before = torch.get_rng_state().clone()
        results = self.wrapper.sample_responses(self.query, self.kv, settings, seed=19)
        torch.testing.assert_close(torch.get_rng_state(), rng_before)
        handle.remove()
        self.assertEqual(calls, [2] + [1] * 9)
        self.assertEqual(len({s['seed'] for s in results}), 3)
        self.assertTrue(all(s['token_count'] == 4 and s['finish_reason'] == 'length' for s in results))
        self.assert_restored()
        resumed = self.wrapper.sample_responses(self.query, self.kv, settings, seed=19, sample_indices=[2])
        self.assertEqual(resumed, results[2:])
        single = self.wrapper.sample_responses(self.query, self.kv, GenerationSettings(temperature=.8, max_new_tokens=4), seed=19)
        self.assertEqual(single, results[:1])
        self.assert_restored()

    def test_greedy_eos_and_callback_exception(self):
        settings = GenerationSettings(max_new_tokens=3)
        first = self.wrapper.sample_responses(self.query, self.kv, settings)
        self.assertEqual(first, self.wrapper.sample_responses(self.query, self.kv, settings))
        self.assertEqual(first, self.wrapper.sample_responses(
            self.query, self.kv, GenerationSettings(
                temperature=.7, top_p=.5, top_k=1, max_new_tokens=3)))
        self.wrapper.gen_kwargs['eos_token_id'] = first[0]['token_ids'][0]
        stopped = self.wrapper.sample_responses(self.query, self.kv, settings)[0]
        self.assertEqual(stopped['finish_reason'], 'eos')
        self.assertEqual(stopped['token_count'], 1)
        self.assertEqual(stopped['text'], '')
        def fail(sample):
            raise RuntimeError('persist failed')
        with self.assertRaisesRegex(RuntimeError, 'persist failed'):
            self.wrapper.sample_responses(self.query, self.kv, settings, on_sample=fail)
        self.assert_restored()

    def test_forward_exception_restores_partially_updated_cache(self):
        def fail(*args):
            raise RuntimeError('layer failed')
        handle = self.wrapper.model.model.layers[1].register_forward_pre_hook(fail)
        with self.assertRaisesRegex(RuntimeError, 'layer failed'):
            self.wrapper.sample_responses(self.query, self.kv, GenerationSettings(max_new_tokens=2))
        handle.remove()
        self.assert_restored()

class HybridSamplingTests(SamplingTests):
    def setUp(self):
        from transformers import Gemma3TextConfig, Gemma3ForCausalLM
        torch.manual_seed(11)
        config = Gemma3TextConfig(vocab_size=19, hidden_size=16, intermediate_size=24,
                                 num_hidden_layers=2, num_attention_heads=2,
                                 num_key_value_heads=1, head_dim=8, sliding_window=16,
                                 sliding_window_pattern=2, eos_token_id=None)
        config._attn_implementation = 'eager'
        self.wrapper = ModelKVzip.__new__(ModelKVzip)
        self.wrapper.model = Gemma3ForCausalLM(config).eval()
        self.wrapper.gen_kwargs = {}
        self.wrapper.decode = lambda ids: ' '.join(map(str, ids.flatten().tolist()))
        self.kv = RetainHybridCache(self.wrapper.model.model, (0, 3), 32)
        self.wrapper(torch.tensor([[2, 3, 4]]), self.kv, update_cache=True)
        self.original = [x.clone() for x in self.kv.key_cache]
        self.original_values = [x.clone() for x in self.kv.value_cache]
        self.original_cur = self.kv._cur_tokens
        self.query = torch.tensor([[5, 6]])

    def assert_restored(self):
        super().assert_restored()
        self.assertEqual(self.kv._cur_tokens, self.original_cur)
        self.assertIsNone(self.kv.backup_sliding_keys)
        self.assertIsNone(self.kv.backup_sliding_values)
        for old, new in zip(self.original_values, self.kv.value_cache):
            torch.testing.assert_close(old, new)


class CacheAwareTinyModel(torch.nn.Module):
    """A one-head causal sum model using real cache updates and pruning masks."""
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(num_hidden_layers=2, num_attention_heads=1,
                                      num_key_value_heads=1, eos_token_id=None)
        self.generation_config = SimpleNamespace(eos_token_id=None)
        self.calls = []
        self.fail_layer = None

    def forward(self, input_ids, past_key_values, **kwargs):
        kv = past_key_values
        self.calls.append(input_ids.shape[1])
        states = input_ids.float().reshape(1, 1, -1, 1).expand(1, self.config.num_key_value_heads, -1, -1)
        for layer in range(2):
            if self.fail_layer == layer:
                raise RuntimeError('partial evict failure')
            if isinstance(kv, RetainHybridCache):
                expanded = states.expand(-1, -1, -1, kv.key_cache[layer].shape[-1])
                positions = kwargs.get('cache_position', torch.arange(kv._seen_tokens, kv._seen_tokens + input_ids.shape[1]))
                keys, _ = kv.update(expanded, expanded, layer, {'cache_position': positions, 'sliding_window': layer not in kv.static_layer_ids})
            else:
                keys, _ = kv.update(states, states, layer)
        if isinstance(kv, (EvictCache, RetainHybridCache)):
            total = keys.sum().long() % 19
        else:
            valid = kv._get_valid(1, keys.shape[-2]) if kv.flatten else torch.ones_like(keys, dtype=torch.bool)
            total = keys.flatten()[valid.flatten()].sum().long() % 19
        logits = torch.zeros((1, input_ids.shape[1], 19))
        logits[:, -1, total] = 3
        return SimpleNamespace(logits=logits)


class PrunedAndEvictTests(unittest.TestCase):
    def make_wrapper(self):
        wrapper = ModelKVzip.__new__(ModelKVzip)
        wrapper.model = CacheAwareTinyModel()
        wrapper.decode = lambda ids: str(ids.tolist())
        wrapper.gen_kwargs = {}
        return wrapper

    def test_first_token_uses_current_prune(self):
        wrapper = self.make_wrapper()
        kv = RetainCache(wrapper.model, (0, 3))
        wrapper.model(torch.tensor([[2, 3, 4]]), past_key_values=kv)
        settings = GenerationSettings(max_new_tokens=1)
        full = wrapper.sample_responses(torch.tensor([[1]]), kv, settings)
        kv.flatten = True
        kv.valid = torch.tensor([[[True, False, False]], [[True, False, False]]])
        pruned = wrapper.sample_responses(torch.tensor([[1]]), kv, settings)
        self.assertEqual(full[0]['token_ids'], [10])
        self.assertEqual(pruned[0]['token_ids'], [3])

    def test_evict_metadata_prompt_reuse_and_exception(self):
        from unittest.mock import patch
        import copy
        def cpu_insert(cache, states, lengths, offsets):
            incoming = states.reshape(len(lengths), -1, states.shape[-1])
            return torch.cat([torch.cat([cache[offsets[h]:offsets[h+1]], incoming[h]]) for h in range(len(lengths))])
        with patch('attention.kvcache.update_flatten_view', cpu_insert, create=True):
            wrapper = self.make_wrapper()
            kv = EvictCache(wrapper.model, (0, 3))
            wrapper.model(torch.tensor([[2, 3, 4]]), past_key_values=kv)
            kv.score = torch.tensor([[[[3., 2., 1.]]], [[[3., 2., 1.]]]])
            kv.prune(.5)
            original_keys = [k.clone() for k in kv.key_cache]
            original_info = copy.deepcopy(kv.info)
            original_q = kv.cu_len_q.clone()
            settings = GenerationSettings(temperature=.8, max_new_tokens=4, num_generations=3)
            wrapper.model.calls.clear()
            samples = wrapper.sample_responses(torch.tensor([[5, 6]]), kv, settings)
            self.assertEqual(wrapper.model.calls, [2] + [1] * 9)
            self.assertEqual(wrapper.sample_responses(torch.tensor([[5, 6]]), kv, settings, sample_indices=[2]), samples[2:])
            wrapper.model.fail_layer = 1
            with self.assertRaisesRegex(RuntimeError, 'partial evict failure'):
                wrapper.sample_responses(torch.tensor([[5, 6]]), kv, settings)
            self.assertEqual(kv._seen_tokens, 3)
            for a, b in zip(original_keys, kv.key_cache):
                torch.testing.assert_close(a, b)
            torch.testing.assert_close(kv.cu_len_q, original_q)
            for name in original_info:
                for a, b in zip(original_info[name], kv.info[name]):
                    torch.testing.assert_close(a, b)

    def test_dynamic_original_tensors_released_after_prompt_prefill(self):
        import weakref
        from unittest.mock import patch

        def cpu_insert(cache, states, lengths, offsets):
            incoming = states.reshape(len(lengths), -1, states.shape[-1])
            return torch.cat([torch.cat([cache[offsets[h]:offsets[h+1]], incoming[h]]) for h in range(len(lengths))])

        for cache_type in (RetainCache, EvictCache):
            with self.subTest(cache_type=cache_type), patch('attention.kvcache.update_flatten_view', cpu_insert, create=True):
                wrapper = self.make_wrapper()
                wrapper.model.config.num_key_value_heads = 2
                wrapper.model.config.num_attention_heads = 2
                kv = cache_type(wrapper.model, (0, 3))
                with torch.inference_mode():
                    wrapper.model(torch.tensor([[2, 3, 4]]), past_key_values=kv)
                if isinstance(kv, EvictCache):
                    for layer in range(2):
                        kv._sample_cache(layer, [torch.tensor([True, False, True]), torch.tensor([False, True, False])])
                original = [tensor.clone() for tensor in kv.key_cache + kv.value_cache]
                refs = [weakref.ref(tensor) for tensor in kv.key_cache + kv.value_cache]
                def check_released(sample):
                    self.assertTrue(all(ref() is None for ref in refs))
                wrapper.sample_responses(torch.tensor([[5, 6]]), kv,
                    GenerationSettings(temperature=.8, max_new_tokens=3, num_generations=2),
                    on_sample=check_released)
                for before, after in zip(original, kv.key_cache + kv.value_cache):
                    torch.testing.assert_close(before, after)
                self.assertEqual(kv._seen_tokens, 3)

    def test_hybrid_sliding_overflow_and_callback_failure(self):
        from transformers import Gemma3TextConfig
        wrapper = self.make_wrapper()
        wrapper.model.config = Gemma3TextConfig(
            vocab_size=19, hidden_size=16, intermediate_size=24,
            num_hidden_layers=2, num_attention_heads=1, num_key_value_heads=1,
            head_dim=8, sliding_window=4, sliding_window_pattern=2, eos_token_id=None)
        kv = RetainHybridCache(wrapper.model, (0, 3), 32)
        wrapper.model(torch.tensor([[2, 3, 4]]), past_key_values=kv,
                      cache_position=torch.arange(3))
        original = [t.clone() for t in kv.key_cache + kv.value_cache]
        settings = GenerationSettings(temperature=.8, max_new_tokens=7, num_generations=3)
        samples = wrapper.sample_responses(torch.tensor([[5, 6]]), kv, settings)
        self.assertEqual(wrapper.sample_responses(torch.tensor([[5, 6]]), kv, settings, sample_indices=[2]), samples[2:])
        def fail(sample):
            raise RuntimeError('callback failure')
        with self.assertRaisesRegex(RuntimeError, 'callback failure'):
            wrapper.sample_responses(torch.tensor([[5, 6]]), kv, settings, on_sample=fail)
        for before, after in zip(original, kv.key_cache + kv.value_cache):
            torch.testing.assert_close(before, after)
        self.assertEqual(kv._seen_tokens, 3)
        self.assertEqual(kv._cur_tokens, 0)
        self.assertIsNone(kv.backup_sliding_keys)


if __name__ == '__main__':
    unittest.main()
