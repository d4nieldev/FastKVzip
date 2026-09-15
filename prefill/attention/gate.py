# ==============================================================================
# Official implementation of "Fast KVzip: Efficient and Accurate LLM Inference with Gated KV Eviction"
# Authors: Jang-Hyun Kim, Dongyoon Han, Sangdoo Yun
# Affiliation: NAVER AI Lab
# Paper: https://arxiv.org/abs/2601.17668
# ==============================================================================
import math
import os

import torch
import torch.nn as nn
from attention.baseline import ExpectedAttentionPress, Head, SnapKV
from huggingface_hub import hf_hub_download
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from utils.func import load_head_score


def load_gate(model, name=""):
    if name == "":
        modules = None
        print("No gate")

    elif name == "expect":
        module = ExpectedAttentionPress(model.model)
        modules = [module for _ in range(model.model.config.num_hidden_layers)]
        print("Expected Attention")

    elif name == "snap":
        module = SnapKV()
        modules = [module for _ in range(model.model.config.num_hidden_layers)]
        print("SnapKV")

    elif name == "head":
        head_score = load_head_score(model.name, 1)
        modules = [Head(score) for score in head_score]
        print("Head-level Eviction")

    else:
        modules = load_fastkvzip(
            model.model.name_or_path, name, device=model.device
        )

    return modules


def is_gate_path(name: str) -> bool:
    """Whether a -g value names a file rather than a released gate.

    Released gates are bare stems that get joined into a repo-relative name, so
    they never contain a separator or the suffix. Deciding by shape rather than
    by existence keeps a mistyped path a loud error instead of a silent fall
    back to the hub.
    """

    return name.endswith(".pt") or os.sep in name


def _layer_state_dicts(payload):
    """Per-layer gate states from either gate file layout."""

    if not isinstance(payload, dict):
        raise ValueError("gate checkpoint must contain a mapping")
    states = payload.get("module")
    if states is not None:
        if not isinstance(states, (list, tuple)) or not states:
            raise ValueError("gate checkpoint 'module' must be a non-empty sequence")
        return list(states)
    # A graph checkpoint has the same {"gate": ...} layout as a gate-only one,
    # and every checkpoint in a run tree is named best.pt or last.pt. Loading
    # one here would score the gate alone and silently discard the mixer's
    # contribution, so refuse rather than return a plausible wrong answer.
    config = payload.get("config")
    if isinstance(config, dict) and config.get("graph_dim") is not None:
        raise ValueError(
            "this checkpoint has a graph mixer, which -g cannot apply; evaluate it "
            "with eval_graph.py --graph-checkpoint instead"
        )
    if payload.get("mixer"):
        raise ValueError("gate checkpoint carries mixer weights that -g cannot apply")
    state = payload.get("gate")
    if not isinstance(state, dict) or not state:
        raise ValueError("gate checkpoint must contain 'module' or 'gate' weights")
    layers = {}
    for key, value in state.items():
        index, _, parameter = key.partition(".")
        if not parameter:
            raise ValueError(f"unexpected gate key {key!r}")
        try:
            layer = int(index)
        except ValueError:
            raise ValueError(f"unexpected gate key {key!r}") from None
        layers.setdefault(layer, {})[parameter] = value
    # Sorting these keys as text would order them 0, 1, 10, 11, 2 and assign
    # every layer's gate to the wrong layer. Every layer has identical shapes,
    # so nothing downstream could catch it.
    if sorted(layers) != list(range(len(layers))):
        raise ValueError("gate checkpoint layer indices must cover 0..n-1")
    return [layers[layer] for layer in sorted(layers)]


class Weight(nn.Module):
    def __init__(
        self,
        index: int,
        input_dim: int,
        output_dim: int,
        nhead: int,
        ngroup: int,
        dtype,
        sink=1,
    ):
        super().__init__()
        self.name = "gate"
        self.index = index
        self.output_dim = output_dim
        self.nhead = nhead
        self.ngroup = ngroup
        self.sink = sink

        self.q_proj = nn.Linear(
            input_dim, nhead * ngroup * output_dim, bias=True, dtype=dtype
        )
        self.k_proj = nn.Linear(input_dim, nhead * output_dim, bias=False, dtype=dtype)
        self.q_norm = Qwen3RMSNorm(output_dim)
        self.k_norm = Qwen3RMSNorm(output_dim)
        self.k_base = nn.Parameter(torch.zeros([nhead, 1, sink, output_dim]))
        self.b = nn.Parameter(torch.zeros([nhead, 1, ngroup], dtype=dtype))

        self.d = math.sqrt(self.output_dim)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.squeeze(0)  # bsz = 1
        # A released gate carries the model dtype, so this is a no-op there. A
        # gate fine-tuned by answer training carries fp32 master weights, and
        # scoring it in fp32 is both required by the projections and what
        # reproduces its training-time scores.
        hidden_states = hidden_states.to(self.q_proj.weight.dtype)
        nseq = hidden_states.shape[0]  # sequence x dim
        hidden_shape = (nseq, self.nhead, -1, self.output_dim)

        queries = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        keys = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        queries = queries.transpose(0, 1).transpose(-1, -2)
        keys = keys.transpose(0, 1)

        # head x seq x 1 x group
        logit = torch.matmul(keys, queries) / self.d + self.b.unsqueeze(2)
        # head x 1 x sink x group
        logit_base = torch.matmul(self.k_base, queries) / self.d
        score = 1 / (1 + torch.exp(logit_base - logit).sum(2, keepdim=True))

        score = score.mean(-1)  # n_head, seq, 1
        return score.squeeze(-1).unsqueeze(0)  # bsz x n_head x seq

    def extra_repr(self):
        # Customize the print output
        repr_str = f"index={self.index}, output_dim={self.output_dim}, nhead={self.nhead}, ngroup={self.ngroup}\n"
        if self.sink != 0:
            repr_str += f"k_base shape: {self.k_base.shape}\n"
        repr_str += f"b shape: {self.b.shape}\n"
        return repr_str


def load_fastkvzip(model_name="Qwen/Qwen3-8B", file_name="fastkvzip", device="cuda"):
    if not model_name:
        raise AssertionError("Model_name is empty. Please check load_gate.")
    state_dict, gate_id = get_gate_weight(model_name, file_name)

    # The gate is scored in the precision it was trained in. That is a no-op for
    # a released gate, whose projections are already the model dtype, and keeps
    # an answer-fine-tuned gate (saved in fp32 master weights) reproducing its
    # own training-time scores rather than drifting by a bf16 rounding.
    dtype = state_dict[0]["q_proj.weight"].dtype
    head_group_outdim, input_dim = state_dict[0]["q_proj.weight"].shape
    head_outdim, _ = state_dict[0]["k_proj.weight"].shape
    output_dim = state_dict[0]["q_norm.weight"].shape[-1]
    nhead = head_outdim // output_dim
    ngroup = head_group_outdim // head_outdim

    # Read the sink from the weights rather than from a "sinkN" filename: the
    # two always agree for a released gate, and a local path may say neither.
    sink = state_dict[0]["k_base"].shape[-2]
    if sink < 1:
        raise ValueError(f"gate {gate_id} has no sink; every score would be 1")

    modules = []
    for l, weight in enumerate(state_dict):
        module = Weight(l, input_dim, output_dim, nhead, ngroup, dtype, sink=sink).to(
            device
        )
        module.load_state_dict(weight)
        modules.append(module)

    print(f"load gate {gate_id} ({module})")
    return modules


def get_gate_id(model_name, file_name="fastkvzip"):
    if file_name == "fastkvzip":
        config = AutoConfig.from_pretrained(model_name)
        if hasattr(config, "text_config"):
            config = config.text_config
        ngroup = config.num_attention_heads // config.num_key_value_heads
        file_name = f"q{ngroup}_dim16_sink16"

    model_name = model_name.split("/")[-1].lower()
    gate_id = os.path.join(model_name, file_name + ".pt")
    return gate_id


def get_gate_weight(model_name, file_name):
    if is_gate_path(file_name):
        path = os.path.expanduser(file_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"gate checkpoint not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return _layer_state_dicts(payload), path

    gate_id = get_gate_id(model_name, file_name)

    try:
        file_path = hf_hub_download(
            repo_id="Jang-Hyun/Fast-KVzip", filename=gate_id, repo_type="model"
        )
    except:
        base_path = "~/FastKVzip"  ## Fix this!
        file_path = os.path.join(base_path, "result_gate", file_name)

    # Load the PyTorch tensor/dictionary
    weights = torch.load(file_path, weights_only=False)["module"]
    return weights, gate_id


if __name__ == "__main__":
    gates = load_gate("qwen3-8b")
