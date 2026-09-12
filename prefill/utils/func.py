import glob
import json
import os
from time import time

import torch


def load_head_score(model_name, ctx_len, noise=1e-4):
    attn_ = []
    paths = f"./utils/head_score/{model_name}-*.pt"
    for path in glob.glob(paths):
        attn = torch.load(path).squeeze().cuda()  # layer x head
        attn_.append(attn)
        print("Load head-score from", path)

    attn = torch.stack(attn_, dim=0).amax(0)
    score = attn.unsqueeze(-1).expand(-1, -1, ctx_len)  # layer x head x seq
    score = score.unsqueeze(1)

    score = score.float()
    score += noise * torch.rand_like(score)  # tie-breaking
    print("Shape:", score.shape)
    return score


def set_gen_length(dataname, model=None):
    if dataname == "longbench_v2":
        from data.longbench_v2 import MAX_NEW_TOKENS

        max_len = MAX_NEW_TOKENS
    elif dataname.startswith("longbench_"):
        from data.longbench import MAX_NEW_TOKENS, parse_longbench_name

        max_len = MAX_NEW_TOKENS[parse_longbench_name(dataname)]
    elif dataname.startswith("ruler_"):
        from data.ruler import parse_ruler_name

        task, _length = parse_ruler_name(dataname)
        max_len = {"niah": 128, "vt": 30, "cwe": 120, "fwe": 50, "qa": 32}[
            task.split("_")[0]
        ]
    elif any(k in dataname for k in ("needle", "_mf")):
        max_len = 48
    elif any(k in dataname for k in ("prefix_suffix",)):
        max_len = 128
    elif any(k in dataname for k in ("squad", "summary")):
        max_len = 256
    elif any(k in dataname for k in ("gsm", "repoqa")):
        max_len = 512
    else:
        max_len = 96

    if model is not None:
        model.gen_kwargs["max_new_tokens"] = max_len
    print(f"set generation length: {max_len} (see utils/func.py)")
    return max_len


def save_result(modelname, args, outputs, idx):
    path = f"./results/{args.data}/{idx}_{modelname}{args.tag}"
    os.makedirs(path, exist_ok=True)

    path = f"{path}/output-{args.level}.json"
    with open(path, "w") as f:
        json.dump(outputs, f, indent=4)
    print(f"Results saved at {path}")


def inplace_softmax(x, dim=-1):
    max_vals, _ = x.max(dim=dim, keepdim=True)
    x.sub_(max_vals)  # For numerical stability
    x.exp_()
    sum_exp = x.sum(dim=dim, keepdim=True)
    x.div_(sum_exp)
    return x


def gmem(text="", print=True):
    _, total_mem = torch.cuda.mem_get_info(0)
    total_mem = total_mem / 1024**3
    allc_mem = torch.cuda.memory_allocated(0) / 1024**3
    msg = f"## {allc_mem:.2f}/{total_mem:.2f} GB, {text}"
    if print:
        print(msg)
    return allc_mem, total_mem


class TimeStamp:

    def __init__(self, verbose=True, precision=1, unit="s"):
        self.verbose = verbose
        self.precision = precision
        self.unit = unit
        self.set()

    def set(self):
        if self.verbose:
            torch.cuda.synchronize()
            self.start = time()

    def elapsed(self, denominator=1.0):
        # example implementation
        val = time() - self.start
        if self.unit == "ms":
            val *= 1000
        return round(val / denominator, self.precision)

    def __call__(self, msg="", denominator=1.0):
        if self.verbose:
            torch.cuda.synchronize()
            allc_mem, total_mem = gmem(print=False)
            tt = self.elapsed(denominator)
            print(
                f"## Time: {tt}{self.unit}. Mem: {allc_mem:.2f}/{total_mem:.2f} GB. {msg}",
            )
            # print(flush=True)
            self.set()
