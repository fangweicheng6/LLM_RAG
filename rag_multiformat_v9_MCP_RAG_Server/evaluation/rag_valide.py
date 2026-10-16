"""直接运行 python rag_valide.py；使用同一个 search.run 评估四项指标。"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
from pathlib import Path
import sys
import time

from utils.config_loader import settings
from evaluation.eval_common import (atomic_text, ensure_artifact_path, load_dataset, retrieval_metrics,
                         sha256_file, summarize, verify_gold_in_index, write_json, write_jsonl)


def evaluate_rows(rows, search_fn, scorer, retrieval_k=0, progress=None):
    results = []
    for index, row in enumerate(rows, 1):
        started = time.monotonic()
        item = {"id": row["id"], "user_input": row["user_input"], "reference": row["reference"],
                "expected_domains": row["expected_domains"],
                "question_type": row.get("question_type", "unknown"),
                "reference_context_ids": row["reference_context_ids"],
                "reviewed": row.get("reviewed", False), "status": "ok", "errors": {},
                "retrieval_recall": None,
                "context_recall": None, "answer_correctness": None}
        try:
            response = search_fn(row["user_input"])
            item.update(retrieval_metrics(
                row, response, retrieval_k,
                router_enabled=settings.rag.router.enabled,
                selector_enabled=settings.rag.context.selector.enabled,
            ))
            item["trace"] = response
        except Exception as exc:
            item["status"] = "pipeline_error"
            item["errors"]["pipeline"] = f"{type(exc).__name__}: {exc}"
        else:
            try:
                scores, errors = scorer.score(row, response)
                item.update(scores)
                item["errors"].update(errors)
                if errors:
                    item["status"] = "metric_error"
            except Exception as exc:
                item["status"] = "metric_error"
                item["errors"]["judge"] = f"{type(exc).__name__}: {exc}"
        item["elapsed_seconds"] = round(time.monotonic() - started, 3)
        results.append(item)
        if progress:
            progress(results, index, len(rows))
    return results


def save_reports(directory, results, metadata):
    settings = metadata.get("settings", {})
    router_enabled = settings.get("router_enabled")
    selector_enabled = settings.get("llm_chunk_selection_enabled")
    summary = {**metadata, **summarize(results,
                                       router_enabled=router_enabled,
                                       selector_enabled=selector_enabled)}
    write_json(directory / "summary.json", summary)
    write_jsonl(directory / "details.jsonl", results)
    buffer = io.StringIO(newline="")
    fields = ["id", "user_input", "question_type", "status", "retrieval_recall", "context_recall",
              "answer_correctness", "elapsed_seconds", "errors"]
    if router_enabled:
        fields.insert(3, "router_f1")
    if selector_enabled:
        fields.insert(-2, "final_context_chunk_recall")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in results:
        safe = {**row, "errors": str(row["errors"]) if row["errors"] else ""}
        # CSV 由 Excel 打开时防止文本被解释为公式；原始问题在 details.jsonl 中完整保留。
        for key in ("id", "user_input", "errors"):
            if isinstance(safe.get(key), str) and safe[key].lstrip().startswith(("=", "+", "-", "@")):
                safe[key] = "'" + safe[key]
        writer.writerow(safe)
    atomic_text(directory / "scores.csv", "\ufeff" + buffer.getvalue())
    lines = ["RAG 评估结果", f"样本: {summary['total']}，全部指标成功: {summary['ok']}，异常: {summary['failed']}"]
    if router_enabled:
        lines.append(f"Router F1 (micro): {summary['router_f1_micro']}")
    else:
        lines.append("Router: OFF")
    for key in ("retrieval_recall", "context_recall", "answer_correctness"):
        score = summary["metrics"][key]
        lines.append(f"{key}: {score['mean']} (有效 {score['valid']}/{summary['total']})")
    if selector_enabled:
        score = summary["metrics"]["final_context_chunk_recall"]
        lines.append(f"final_context_chunk_recall: {score['mean']} (有效 {score['valid']}/{summary['total']})")
    else:
        lines.append("LLM Chunk Selector: OFF")
    if "by_question_type" in summary:
        lines.append("")
        lines.append("按 question_type 分类评估:")
        for qtype, block in summary["by_question_type"].items():
            lines.append(f"{qtype}: {block['count']} 条")
            for metric, value in block["metrics"].items():
                lines.append(f"  {metric}: {value}")
    if "weighted_overall" in summary:
        lines.append("")
        lines.append("比例加权总评估:")
        for metric, value in summary["weighted_overall"].items():
            lines.append(f"{metric}: {value}")
    lines.extend(["", "异常分数为 null，不作为 0 或 1；任何异常导致命令退出码为 1。",
                  "Retrieval Recall 计算首轮候选池；Context Recall 计算最终送入答案生成器的上下文。",
                  "reviewed=false 的合成样本仅作为候选真值，请复核后再解读正式准确率。"])
    atomic_text(directory / "summary.txt", "\n".join(lines) + "\n")
    return summary


def count_index_chunks(path):
    """Count usable chunk IDs in the current index manifest without hydrating source text."""
    import json
    seen = set()
    path = Path(path)
    if not path.exists():
        return 0
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            cid = row.get("chunk_id")
            if cid and row.get("domain") in settings.domains.allowed:
                seen.add(cid)
        except Exception:
            continue
    return len(seen)


def main(argv=None):
    parser = argparse.ArgumentParser(description="评估 Router F1、Retrieval Recall、Context Recall、Answer Correctness")
    parser.add_argument("--dataset", type=Path, default=settings.evaluation.paths.dataset)
    parser.add_argument("--output-dir", type=Path, help="默认建立含时间戳的新报告目录，禁止覆盖已有目录")
    parser.add_argument("--limit", type=int, help="仅评估前 N 条，用于小规模联调")
    parser.add_argument("--retrieval-k", type=int, default=settings.evaluation.runtime.retrieval_k)
    parser.add_argument("--require-reviewed", action="store_true", help="遇到未经人工确认的样本时报错")
    parser.add_argument("--check-only", action="store_true", help="只验证数据集和向量库真值定位，不调用 LLM")
    parser.add_argument("--verbose", action="store_true", help="显示原检索流程日志")
    args = parser.parse_args(argv)
    try:
        if args.retrieval_k < 0 or (args.limit is not None and args.limit <= 0):
            raise ValueError("retrieval-k 必须 >= 0，limit 必须 > 0")
        rows = load_dataset(args.dataset)
        if args.limit:
            rows = rows[:args.limit]
        unreviewed = sum(not row.get("reviewed", False) for row in rows)
        if unreviewed and args.require_reviewed:
            raise ValueError(f"存在 {unreviewed} 条未人工确认的样本")
        from pipeline.retrieval.retriever import get_client, load_embedding_model, run
        client = get_client()
        verified = verify_gold_in_index(rows, client)
        total_gold_refs = sum(len(row.get("reference_context_ids", [])) for row in rows)
        index_chunks = count_index_chunks(settings.paths.chunk_jsonl)
        coverage = (verified / index_chunks) if index_chunks else 0.0
        uniqueness = (verified / total_gold_refs) if total_gold_refs else 0.0
        print(
            f"数据集检查通过: {len(rows)} 个问题；"
            f"Gold引用总数={total_gold_refs}；唯一真值chunk={verified}/{index_chunks} "
            f"(知识库覆盖率={coverage:.1%}, 引用唯一率={uniqueness:.1%})；"
            f"未复核={unreviewed}"
        )
        if args.check_only:
            print("未调用 LLM。")
            return 0
        directory = ensure_artifact_path(args.output_dir or settings.evaluation.paths.dir / (
            "run_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
        if directory.exists():
            raise ValueError(f"报告目录已存在，请指定新目录: {directory}")
        from llm.factory import resolve_llm_model
        from evaluation.ragas_adapter import RagasScorer, ProjectRagasEmbeddings
        model = load_embedding_model()
        judge_embeddings = ProjectRagasEmbeddings(
            model=model if settings.evaluation.llm.embedding_model == settings.model.embedding.name else None)
        scorer = RagasScorer(embeddings=judge_embeddings)
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(), "ragas_version": "0.4.3",
            "dataset": str(args.dataset.resolve()), "dataset_sha256": sha256_file(args.dataset),
            "sample_count": len(rows), "unreviewed_count": unreviewed, "complete": False,
            "settings": {
                "router_enabled": settings.rag.router.enabled,
                "llm_chunk_selection_enabled": settings.rag.context.selector.enabled,
                # Disabled means the metric is intentionally not computed. Treating
                # all domains as a prediction would create the misleading constant 1/3.
                "router_f1_mode": "raw_router_domains" if settings.rag.router.enabled else "disabled",
                "selector_diagnostic_mode": (
                    "final_context_chunk_recall"
                    if settings.rag.context.selector.enabled else "disabled"
                ),
                "retrieval_k": args.retrieval_k,
                "candidate_pool_size_per_domain": settings.rag.retrieval.candidate_pool_size_per_domain,
                "vector_context_top_k": settings.rag.retrieval.top_k,
                "expansion_rounds": settings.rag.context.expander.max_rounds,
                "answer_model": resolve_llm_model("generator"), "judge_model": resolve_llm_model("evaluation"),
                "embedding_model": settings.model.embedding.name,
                "judge_embedding_model": settings.evaluation.llm.embedding_model,
                "metric_parse_retries": settings.evaluation.runtime.metric_parse_retries,
                "answer_correctness_weights": settings.evaluation.runtime.answer_correctness_weights,
            },
        }
        def search_fn(question):
            return run(question, client=client, model=model, quiet=not args.verbose,
                       allow_empty=True, strict=True)
        def progress(results, index, total):
            save_reports(directory, results, metadata)
            print(f"[{index}/{total}] {results[-1]['id']}: {results[-1]['status']}")
        results = evaluate_rows(rows, search_fn, scorer, args.retrieval_k, progress)
        metadata["complete"] = True
        summary = save_reports(directory, results, metadata)
        print((directory / "summary.txt").read_text(encoding="utf-8"))
        print(f"完整报告: {directory}")
        return 1 if summary["failed"] else 0
    except Exception as exc:
        print(f"评估失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        if isinstance(exc, (ImportError, ModuleNotFoundError)):
            print("请安装原项目依赖及 python -m pip install -r requirements_eval.txt", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
