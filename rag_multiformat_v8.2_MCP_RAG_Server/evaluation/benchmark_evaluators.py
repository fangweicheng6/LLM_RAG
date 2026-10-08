"""Router / Retrieval / Generation 三个专项 benchmark 评估器。

E2E 继续复用 rag_valide.py，保证正式端到端指标只有一套实现。
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import time
from typing import Iterable

from utils.config_loader import settings
from evaluation.eval_common import ensure_artifact_path, set_scores, sha256_file, write_json, write_jsonl


def _load_jsonl(path: Path, required: Iterable[str]) -> list[dict]:
    path = Path(path)
    rows = []
    seen = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} 第 {line_no} 行 JSON 无效: {exc}") from exc
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"{path} 第 {line_no} 行缺少字段: {missing}")
        if row["id"] in seen:
            raise ValueError(f"{path} 样本 ID 重复: {row['id']}")
        seen.add(row["id"])
        rows.append(row)
    if not rows:
        raise ValueError(f"benchmark 数据为空: {path}")
    return rows


def _mean(values):
    values = [float(x) for x in values if x is not None]
    return sum(values) / len(values) if values else None


def _metric_block(results: list[dict], metric_names: list[str]) -> dict:
    block = {}
    for name in metric_names:
        values = [row.get(name) for row in results if row.get(name) is not None]
        block[name] = {
            "mean": _mean(values),
            "valid": len(values),
            "missing": len(results) - len(values),
        }
    return block


def _by_question_type(results: list[dict], metric_names: list[str]) -> dict:
    grouped = {}
    for row in results:
        qtype = row.get("question_type", "unknown")
        grouped.setdefault(qtype, []).append(row)
    output = {}
    for qtype, rows in grouped.items():
        output[qtype] = {
            "count": len(rows),
            "metrics": {name: _mean([r.get(name) for r in rows]) for name in metric_names},
        }
    return output


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        safe = {**row, "errors": json.dumps(row.get("errors", {}), ensure_ascii=False)}
        for key in fields:
            if isinstance(safe.get(key), (dict, list)):
                safe[key] = json.dumps(safe[key], ensure_ascii=False)
            if isinstance(safe.get(key), str) and safe[key].lstrip().startswith(("=", "+", "-", "@")):
                safe[key] = "'" + safe[key]
        writer.writerow(safe)
    path.write_text("\ufeff" + buffer.getvalue(), encoding="utf-8")


def _prepare_output(directory: Path) -> Path:
    directory = ensure_artifact_path(directory)
    if directory.exists():
        raise ValueError(f"报告目录已存在，请使用新目录: {directory}")
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _finalize(directory: Path, results: list[dict], summary: dict, fields: list[str], text_lines: list[str]):
    write_json(directory / "summary.json", summary)
    write_jsonl(directory / "details.jsonl", results)
    _write_csv(directory / "scores.csv", results, fields)
    (directory / "summary.txt").write_text("\n".join(text_lines).rstrip() + "\n", encoding="utf-8")
    return summary


def evaluate_router(dataset: Path, output_dir: Path, limit: int | None = None, llm=None) -> dict:
    rows = _load_jsonl(dataset, ("id", "user_input", "expected_domains"))
    if limit:
        rows = rows[:limit]
    directory = _prepare_output(output_dir)

    from llm import create_llm_client
    from pipeline.routing.router import classify_domains
    llm = llm or create_llm_client(role="router")

    results = []
    for index, row in enumerate(rows, 1):
        started = time.monotonic()
        item = {
            "id": row["id"],
            "user_input": row["user_input"],
            "question_type": row.get("question_type", "unknown"),
            "expected_domains": row["expected_domains"],
            "predicted_domains": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "exact_match": None,
            "status": "ok",
            "errors": {},
        }
        try:
            predicted, reason = classify_domains(row["user_input"], llm)
            counts = set_scores(predicted, row["expected_domains"])
            item.update({
                "predicted_domains": predicted,
                "router_reason": reason,
                "precision": counts["precision"],
                "recall": counts["recall"],
                "f1": counts["f1"],
                "exact_match": float(set(predicted) == set(row["expected_domains"])),
                "tp": counts["tp"], "fp": counts["fp"], "fn": counts["fn"],
            })
        except Exception as exc:
            item["status"] = "error"
            item["errors"]["router"] = f"{type(exc).__name__}: {exc}"
        item["elapsed_seconds"] = round(time.monotonic() - started, 3)
        results.append(item)
        write_jsonl(directory / "details.jsonl", results)
        print(f"[{index}/{len(rows)}] {row['id']}: {item['status']}")

    counts = [r for r in results if r.get("tp") is not None]
    tp, fp, fn = (sum(r[key] for r in counts) for key in ("tp", "fp", "fn")) if counts else (0, 0, 0)
    micro_precision = tp / (tp + fp) if (tp + fp) else 0.0
    micro_recall = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = 2 * tp / (2 * tp + fp + fn) if counts else None
    metric_names = ["precision", "recall", "f1", "exact_match"]
    summary = {
        "benchmark": "router",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(Path(dataset).resolve()),
        "dataset_sha256": sha256_file(dataset),
        "total": len(results),
        "ok": sum(r["status"] == "ok" for r in results),
        "failed": sum(r["status"] != "ok" for r in results),
        "metrics": _metric_block(results, metric_names),
        "micro": {"precision": micro_precision, "recall": micro_recall, "f1": micro_f1,
                  "tp": tp, "fp": fp, "fn": fn},
        "by_question_type": _by_question_type(results, metric_names),
        "settings": {"runtime_router_enabled": settings.rag.router.enabled,
                     "note": "专项 Router benchmark 直接测试 Domain Router，不受 E2E 开关影响。"},
    }
    text = [
        "Router Benchmark",
        f"样本: {summary['total']}，成功: {summary['ok']}，异常: {summary['failed']}",
        f"Precision (micro): {micro_precision}",
        f"Recall (micro): {micro_recall}",
        f"F1 (micro): {micro_f1}",
        f"Exact Match Accuracy: {summary['metrics']['exact_match']['mean']}",
        f"Macro F1: {summary['metrics']['f1']['mean']}",
    ]
    return _finalize(
        directory, results, summary,
        ["id", "user_input", "question_type", "status", "expected_domains", "predicted_domains",
         "precision", "recall", "f1", "exact_match", "elapsed_seconds", "errors"],
        text,
    )


def _ranking_scores(candidate_ids: list[str], gold_ids: list[str], ks: tuple[int, ...]) -> dict:
    gold = set(gold_ids)
    if not gold:
        raise ValueError("reference_context_ids 不能为空")
    first_rank = next((i for i, cid in enumerate(candidate_ids, 1) if cid in gold), None)
    scores = {
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "first_relevant_rank": first_rank,
        "candidate_pool_recall": len(set(candidate_ids) & gold) / len(gold),
    }
    for k in ks:
        top = candidate_ids[:k]
        hit_count = len(set(top) & gold)
        # 标准 Precision@K：不足 K 个结果时，缺失位置按未命中处理。
        scores[f"recall@{k}"] = hit_count / len(gold)
        scores[f"precision@{k}"] = hit_count / k
        scores[f"hit_rate@{k}"] = float(hit_count > 0)
    return scores


def evaluate_retrieval(dataset: Path, output_dir: Path, limit: int | None = None,
                       ks: tuple[int, ...] | None = None, verbose: bool = False,
                       llm=None, client=None, model=None) -> dict:
    rows = _load_jsonl(dataset, ("id", "user_input", "reference_context_ids"))
    if limit:
        rows = rows[:limit]
    ks = tuple(sorted(set(ks or settings.evaluation.benchmark.retrieval_ks)))
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("Retrieval K 必须是正整数")
    directory = _prepare_output(output_dir)

    from llm import create_llm_client
    from pipeline.retrieval.retriever import get_client, load_embedding_model, run_retrieval
    llm = llm or create_llm_client(role="router")
    client = client or get_client()
    model = model or load_embedding_model()

    # 只做 ID 存在性校验；专项数据不包含正文，因此不调用 verify_gold_in_index。
    available_ids = set()
    try:
        from pipeline.retrieval.retriever import get_available_domains
        for domain in get_available_domains(client):
            collection = client.get_collection(name=settings.paths.collection_prefix + domain, embedding_function=None)
            got = collection.get(include=["metadatas"])
            available_ids.update(got.get("ids", []))
    except Exception as exc:
        raise RuntimeError(f"无法校验 Retrieval Gold ID: {exc}") from exc
    wanted = {cid for row in rows for cid in row["reference_context_ids"]}
    missing = sorted(wanted - available_ids)
    if missing:
        raise ValueError(f"Retrieval benchmark 有 {len(missing)} 个 Gold chunk 不在当前向量库: {missing[:5]}")

    results = []
    for index, row in enumerate(rows, 1):
        started = time.monotonic()
        item = {
            "id": row["id"], "user_input": row["user_input"],
            "question_type": row.get("question_type", "unknown"),
            "reference_context_ids": row["reference_context_ids"],
            "status": "ok", "errors": {},
        }
        try:
            trace = run_retrieval(
                row["user_input"], llm=llm, client=client, model=model,
                quiet=not verbose, allow_empty=True, strict=True,
            )
            candidate_ids = [str(x["chunk_id"]) for x in trace.get("candidate_pool", [])]
            item.update(_ranking_scores(candidate_ids, row["reference_context_ids"], ks))
            item.update({
                "candidate_count": len(candidate_ids),
                "candidate_ids": candidate_ids,
                "router_domains": trace.get("router_domains", []),
                "retrieval_domains": trace.get("domains", []),
            })
        except Exception as exc:
            item["status"] = "error"
            item["errors"]["retrieval"] = f"{type(exc).__name__}: {exc}"
        item["elapsed_seconds"] = round(time.monotonic() - started, 3)
        results.append(item)
        write_jsonl(directory / "details.jsonl", results)
        print(f"[{index}/{len(rows)}] {row['id']}: {item['status']}")

    metric_names = ["candidate_pool_recall", "mrr"]
    for k in ks:
        metric_names.extend([f"recall@{k}", f"precision@{k}", f"hit_rate@{k}"])
    summary = {
        "benchmark": "retrieval",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(Path(dataset).resolve()),
        "dataset_sha256": sha256_file(dataset),
        "total": len(results),
        "ok": sum(r["status"] == "ok" for r in results),
        "failed": sum(r["status"] != "ok" for r in results),
        "metrics": _metric_block(results, metric_names),
        "by_question_type": _by_question_type(results, metric_names),
        "settings": {
            "ks": list(ks),
            "router_enabled": settings.rag.router.enabled,
            "reranker_enabled": settings.rag.retrieval.rerank.enabled,
            "rerank_top_k": settings.rag.retrieval.rerank.top_k,
            "multi_hop_enabled": settings.rag.retrieval.multi_hop.enabled,
            "candidate_pool_size_per_domain": settings.rag.retrieval.candidate_pool_size_per_domain,
        },
    }
    text = [
        "Retrieval Benchmark",
        f"样本: {summary['total']}，成功: {summary['ok']}，异常: {summary['failed']}",
        f"Candidate Pool Recall: {summary['metrics']['candidate_pool_recall']['mean']}",
        f"MRR: {summary['metrics']['mrr']['mean']}",
    ]
    for k in ks:
        text.extend([
            f"Recall@{k}: {summary['metrics'][f'recall@{k}']['mean']}",
            f"Precision@{k}: {summary['metrics'][f'precision@{k}']['mean']}",
            f"Hit Rate@{k}: {summary['metrics'][f'hit_rate@{k}']['mean']}",
        ])
    return _finalize(
        directory, results, summary,
        ["id", "user_input", "question_type", "status", "candidate_count", "first_relevant_rank",
         "candidate_pool_recall", "mrr"] +
        [name for k in ks for name in (f"recall@{k}", f"precision@{k}", f"hit_rate@{k}")] +
        ["reference_context_ids", "candidate_ids", "router_domains", "retrieval_domains", "elapsed_seconds", "errors"],
        text,
    )


def _gold_context_items(row: dict) -> list[dict]:
    contexts = list(row.get("reference_contexts", []))
    ids = list(row.get("reference_context_ids", []))
    if ids and len(ids) != len(contexts):
        raise ValueError(f"样本 {row['id']} 的 reference_context_ids 与 reference_contexts 数量不一致")
    if not ids:
        ids = [f"gold_{i}" for i in range(1, len(contexts) + 1)]
    sources = {str(x.get("chunk_id")): x for x in row.get("reference_sources", []) if isinstance(x, dict)}
    items = []
    for cid, text in zip(ids, contexts):
        source = sources.get(cid, {})
        items.append({
            "chunk_id": cid,
            "document": text,
            "metadata": {
                "source_file": source.get("source_file", "gold_context"),
                "section_path": source.get("section_path", ""),
                "relative_path": source.get("relative_path", ""),
            },
        })
    return items


def evaluate_generation(dataset: Path, output_dir: Path, limit: int | None = None,
                        llm=None, scorer=None) -> dict:
    rows = _load_jsonl(dataset, ("id", "user_input", "reference", "reference_contexts"))
    if limit:
        rows = rows[:limit]
    directory = _prepare_output(output_dir)

    from pipeline.generation.generator import generate_answer
    from llm import create_llm_client, resolve_llm_model
    llm = llm or create_llm_client(role="generator")
    if scorer is None:
        from evaluation.ragas_adapter import GenerationRagasScorer, ProjectRagasEmbeddings
        scorer = GenerationRagasScorer(embeddings=ProjectRagasEmbeddings())

    results = []
    for index, row in enumerate(rows, 1):
        started = time.monotonic()
        item = {
            "id": row["id"], "user_input": row["user_input"], "reference": row["reference"],
            "question_type": row.get("question_type", "unknown"),
            "gold_context_count": len(row["reference_contexts"]),
            "answer": None, "answer_correctness": None, "faithfulness": None,
            "answer_relevancy": None, "status": "ok", "errors": {},
        }
        try:
            contexts = _gold_context_items(row)
            answer, sources = generate_answer(
                query=row["user_input"], selected_contexts=contexts, llm=llm,
                context_sufficient=True, missing_information=[],
            )
            item["answer"] = answer
            item["answer_sources"] = sources
            scores, errors = scorer.score(row, answer, row["reference_contexts"])
            item.update(scores)
            item["errors"].update(errors)
            if errors:
                item["status"] = "metric_error"
        except Exception as exc:
            item["status"] = "error"
            item["errors"]["generation"] = f"{type(exc).__name__}: {exc}"
        item["elapsed_seconds"] = round(time.monotonic() - started, 3)
        results.append(item)
        write_jsonl(directory / "details.jsonl", results)
        print(f"[{index}/{len(rows)}] {row['id']}: {item['status']}")

    metric_names = ["answer_correctness", "faithfulness", "answer_relevancy"]
    summary = {
        "benchmark": "generation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(Path(dataset).resolve()),
        "dataset_sha256": sha256_file(dataset),
        "total": len(results),
        "ok": sum(r["status"] == "ok" for r in results),
        "failed": sum(r["status"] != "ok" for r in results),
        "metrics": _metric_block(results, metric_names),
        "by_question_type": _by_question_type(results, metric_names),
        "settings": {
            "input_context": "gold_context_only",
            "answer_model": resolve_llm_model("generator"),
            "judge_model": resolve_llm_model("evaluation"),
            "judge_embedding_model": settings.evaluation.llm.embedding_model,
            "answer_correctness_weights": settings.evaluation.runtime.answer_correctness_weights,
        },
    }
    text = [
        "Generation Benchmark (Gold Context -> Answer Generator)",
        f"样本: {summary['total']}，成功: {summary['ok']}，异常: {summary['failed']}",
        f"Answer Correctness: {summary['metrics']['answer_correctness']['mean']}",
        f"Faithfulness: {summary['metrics']['faithfulness']['mean']}",
        f"Answer Relevancy: {summary['metrics']['answer_relevancy']['mean']}",
        "说明: 该专项绕过 Router/Retrieval/Selector，仅测试给定正确上下文时的生成质量。",
    ]
    return _finalize(
        directory, results, summary,
        ["id", "user_input", "question_type", "status", "gold_context_count", "answer",
         "answer_correctness", "faithfulness", "answer_relevancy", "elapsed_seconds", "errors"],
        text,
    )
