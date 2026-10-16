"""统一运行 Router / Retrieval / Generation / E2E 四类 benchmark。

示例：
    python run_benchmark_evaluation.py --type all
    python run_benchmark_evaluation.py --type retrieval --retrieval-ks 1,3,5,10
    python run_benchmark_evaluation.py --type generation --limit 10
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from utils.config_loader import settings
from evaluation.benchmark_splitter import split_benchmarks
from evaluation.eval_common import ensure_artifact_path, write_json


def _parse_ks(value: str) -> tuple[int, ...]:
    try:
        ks = tuple(sorted({int(x.strip()) for x in value.split(",") if x.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("retrieval-ks 必须是逗号分隔的整数") from exc
    if not ks or any(k <= 0 for k in ks):
        raise argparse.ArgumentTypeError("retrieval-ks 必须全部 > 0")
    return ks


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _summary_lines(summaries: dict, output_root: Path) -> list[str]:
    lines = ["RAG Benchmark Summary", f"报告目录: {output_root}"]
    if "router" in summaries and summaries["router"].get("micro"):
        s = summaries["router"]
        lines.extend([
            "",
            "[Router]",
            f"samples={s.get('total')} failed={s.get('failed')}",
            f"precision_micro={s['micro'].get('precision')}",
            f"recall_micro={s['micro'].get('recall')}",
            f"f1_micro={s['micro'].get('f1')}",
            f"exact_match={s.get('metrics', {}).get('exact_match', {}).get('mean')}",
        ])
    if "retrieval" in summaries and summaries["retrieval"].get("metrics"):
        s = summaries["retrieval"]
        ks = s.get("settings", {}).get("ks", [])
        lines.extend([
            "",
            "[Retrieval]",
            f"samples={s.get('total')} failed={s.get('failed')}",
            f"candidate_pool_recall={s['metrics'].get('candidate_pool_recall', {}).get('mean')}",
            f"mrr={s['metrics'].get('mrr', {}).get('mean')}",
        ])
        for k in ks:
            lines.append(f"recall@{k}={s['metrics'].get(f'recall@{k}', {}).get('mean')}")
            lines.append(f"hit_rate@{k}={s['metrics'].get(f'hit_rate@{k}', {}).get('mean')}")
    if "generation" in summaries and summaries["generation"].get("metrics"):
        s = summaries["generation"]
        lines.extend([
            "",
            "[Generation]",
            f"samples={s.get('total')} failed={s.get('failed')}",
            f"answer_correctness={s['metrics'].get('answer_correctness', {}).get('mean')}",
            f"faithfulness={s['metrics'].get('faithfulness', {}).get('mean')}",
            f"answer_relevancy={s['metrics'].get('answer_relevancy', {}).get('mean')}",
        ])
    if "e2e" in summaries and summaries["e2e"].get("metrics"):
        s = summaries["e2e"]
        lines.extend([
            "",
            "[E2E]",
            f"samples={s.get('total')} failed={s.get('failed')}",
        ])
        if s.get("router_f1_micro") is not None:
            lines.append(f"router_f1_micro={s.get('router_f1_micro')}")
        for name in ("retrieval_recall", "context_recall", "final_context_chunk_recall", "answer_correctness"):
            if name in s.get("metrics", {}):
                lines.append(f"{name}={s['metrics'][name].get('mean')}")
    for name, s in summaries.items():
        if s.get("runner_error"):
            lines.extend(["", f"[{name}] RUNNER ERROR", s["runner_error"]])
    return lines


def main(argv=None):
    p = argparse.ArgumentParser(description="四类 RAG benchmark 统一评估入口")
    p.add_argument("--type", choices=["all", "router", "retrieval", "generation", "e2e"], default="all")
    p.add_argument("--dataset", type=Path, default=settings.evaluation.paths.dataset,
                   help="统一评估集；默认先刷新 benchmarks/*.jsonl")
    p.add_argument("--benchmark-dir", type=Path, default=settings.evaluation.paths.benchmark_dir)
    p.add_argument("--output-dir", type=Path,
                   help="默认 data_index_only/dataset/benchmark_results/run_时间戳")
    p.add_argument("--no-refresh-benchmarks", action="store_true",
                   help="不从统一评估集重新生成四类 benchmark JSONL")
    p.add_argument("--limit", type=int, help="每个 benchmark 仅评估前 N 条，用于联调")
    p.add_argument("--retrieval-ks", type=_parse_ks,
                   default=tuple(settings.evaluation.benchmark.retrieval_ks),
                   help="Retrieval 的 K，例如 1,3,5,10")
    p.add_argument("--e2e-retrieval-k", type=int, default=settings.evaluation.runtime.retrieval_k,
                   help="E2E 旧指标 retrieval_recall 的截断 K；0=整个首轮候选池")
    p.add_argument("--require-reviewed", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    if args.limit is not None and args.limit <= 0:
        p.error("--limit 必须 > 0")
    if args.e2e_retrieval_k < 0:
        p.error("--e2e-retrieval-k 必须 >= 0")

    types = ["router", "retrieval", "generation", "e2e"] if args.type == "all" else [args.type]
    benchmark_dir = Path(args.benchmark_dir).resolve()

    print("Benchmark mode:", args.type)
    if not args.no_refresh_benchmarks:
        manifest = split_benchmarks(args.dataset, benchmark_dir)
        print("Benchmark datasets refreshed:", manifest["counts"])
    else:
        manifest_path = benchmark_dir / "benchmark_manifest.json"
        print("Benchmark datasets: reuse existing files")
        if not manifest_path.exists():
            print("[WARN] benchmark_manifest.json 不存在，无法确认 benchmark 与统一评估集是否同步。")

    paths = {
        "router": benchmark_dir / "router_test.jsonl",
        "retrieval": benchmark_dir / "retrieval_test.jsonl",
        "generation": benchmark_dir / "generation_test.jsonl",
        "e2e": benchmark_dir / "e2e_test.jsonl",
    }
    for t in types:
        if not paths[t].exists():
            print(f"缺少 benchmark 文件: {paths[t]}", file=sys.stderr)
            return 1

    output_root = ensure_artifact_path(args.output_dir or Path(settings.evaluation.paths.result_dir) / (
        "run_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ))
    if output_root.exists():
        print(f"输出目录已存在，请指定新目录: {output_root}", file=sys.stderr)
        return 1
    output_root.mkdir(parents=True, exist_ok=False)

    summaries = {}
    any_failure = False

    for t in types:
        print(f"\n====== {t} benchmark ======")
        try:
            if t == "router":
                from evaluation.benchmark_evaluators import evaluate_router
                summaries[t] = evaluate_router(paths[t], output_root / t, limit=args.limit)
            elif t == "retrieval":
                from evaluation.benchmark_evaluators import evaluate_retrieval
                summaries[t] = evaluate_retrieval(
                    paths[t], output_root / t, limit=args.limit,
                    ks=args.retrieval_ks, verbose=args.verbose,
                )
            elif t == "generation":
                from evaluation.benchmark_evaluators import evaluate_generation
                summaries[t] = evaluate_generation(paths[t], output_root / t, limit=args.limit)
            else:
                # 以模块方式从项目根目录运行，保证 utils/evaluation 等绝对导入可用。
                project_root = Path(__file__).resolve().parent.parent
                cmd = [
                    sys.executable, "-m", "evaluation.rag_valide",
                    "--dataset", str(paths[t]),
                    "--output-dir", str(output_root / "e2e"),
                    "--retrieval-k", str(args.e2e_retrieval_k),
                ]
                if args.limit:
                    cmd += ["--limit", str(args.limit)]
                if args.require_reviewed:
                    cmd += ["--require-reviewed"]
                if args.verbose:
                    cmd += ["--verbose"]
                completed = subprocess.run(cmd, cwd=project_root, check=False)
                summaries[t] = _read_json(output_root / "e2e" / "summary.json") or {
                    "benchmark": "e2e",
                    "runner_error": f"rag_valide.py 未生成 summary.json，退出码={completed.returncode}",
                    "failed": 1,
                }
                if completed.returncode != 0:
                    any_failure = True
            if summaries[t].get("failed", 0):
                any_failure = True
        except Exception as exc:
            any_failure = True
            summaries[t] = {
                "benchmark": t,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "runner_error": f"{type(exc).__name__}: {exc}",
                "failed": 1,
            }
            print(f"{t} benchmark 失败: {type(exc).__name__}: {exc}", file=sys.stderr)

    overall = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.type,
        "source_dataset": str(Path(args.dataset).resolve()),
        "benchmark_dir": str(benchmark_dir),
        "output_dir": str(output_root),
        "feature_flags": {
            "router_enabled": settings.rag.router.enabled,
            "llm_chunk_selection_enabled": settings.rag.context.selector.enabled,
            "reranker_enabled": settings.rag.retrieval.rerank.enabled,
            "multi_hop_retriever_enabled": settings.rag.retrieval.multi_hop.enabled,
        },
        "summaries": summaries,
        "complete": not any_failure,
    }
    write_json(output_root / "benchmark_summary.json", overall)
    lines = _summary_lines(summaries, output_root)
    (output_root / "benchmark_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\n完整 benchmark 报告: {output_root}")
    return 1 if any_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
