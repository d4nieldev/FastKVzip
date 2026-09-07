# ==============================================================================
# Benchmark evaluation with KV eviction after prefill
# ==============================================================================
from collections import defaultdict
from data.benchmarks import get_data_list


def set_ratios():
    ratios = [0.75, 0.5, 0.4, 0.3, 0.2]
    return ratios


if __name__ == "__main__":
    from args import args
    from attention.gate import load_gate
    from model import ModelKVzip

    from data import DataWrapper, load_dataset_all
    from utils import Evaluator, TimeStamp, save_result, set_gen_length

    if args.gate_path_or_name:
        args.tag += f"_w{args.window_size}"
        print(f"tag: {args.tag}")

    args.kv_type = "retain"  # RetainCache enables efficient evaluation across multiple compression ratios with a single prefilling.
    model = ModelKVzip(args.model, args.kv_type, args.gate_path_or_name)

    for args.data in get_data_list(args.data, model.name):
        dataset = load_dataset_all(args.data, model.tokenizer)  # list of data
        dataset = DataWrapper(args.data, dataset, model)
        set_gen_length(args.data, model)

        tt = TimeStamp(True)
        max_idx = min(args.idx + args.num, len(dataset))
        print("=" * 80, f"\nStart evaluation with {args.idx}~{max_idx} samples")

        for data_idx in range(args.idx, max_idx):
            kv = dataset.prefill_context(
                data_idx,
                window_size=args.window_size,
                do_score=True,
            )
            inputs, info = dataset.generate_answer(data_idx, kv, prob=False)
            eval = Evaluator(model, inputs, info)

            outputs = defaultdict(list)
            for ratio in set_ratios():
                thres, ratio_true = kv.prune(ratio, args.level)
                results = eval(kv, generate=True)  # generation

                for fmt, v in results.items():
                    outputs[fmt].append(
                        [[ratio, round(ratio_true, 4), round(thres, 4)], v]
                    )

            save_result(model.name, args, outputs, data_idx)

            tt(f"[{args.data}-{data_idx}]\n")
            del kv, inputs, info, eval
        print("Finished.")
