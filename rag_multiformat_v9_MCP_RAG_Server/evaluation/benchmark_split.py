"""兼容旧命令：内部统一调用 benchmark_splitter.py。"""
from pathlib import Path
import argparse

from utils.config_loader import settings
from evaluation.benchmark_splitter import split_benchmarks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=settings.evaluation.paths.dataset)
    p.add_argument("--output", type=Path, default=settings.evaluation.paths.benchmark_dir)
    args = p.parse_args()
    manifest = split_benchmarks(args.dataset, args.output)
    print(f"benchmark 已生成: {args.output}")
    print("counts:", manifest["counts"])


if __name__ == "__main__":
    main()
