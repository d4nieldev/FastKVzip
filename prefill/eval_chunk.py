"""Benchmark evaluation with native chunked-prefill eviction."""

from eval import main


if __name__ == "__main__":
    main(chunked=True)
