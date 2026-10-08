"""数据集、溯源校验和确定性指标；本模块仅依赖标准库。"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

from utils.config_loader import settings


def ensure_artifact_path(path):
    path = Path(path).resolve()
    for protected in (settings.paths.chroma_dir, settings.paths.source_store_dir):
        if path.is_relative_to(Path(protected).resolve()):
            raise ValueError("评估数据/报告不能写入 Chroma 或原文快照目录")
    if path == Path(settings.paths.chunk_jsonl).resolve():
        raise ValueError("不能覆盖索引清单")
    return path


def atomic_text(path, text):
    path = ensure_artifact_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                     dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                              for row in rows))


def sha256_file(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def validate_rows(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("评估数据集必须为非空列表/JSONL")
    seen, questions = set(), set()
    for number, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"样本 {number} 必须是 JSON 对象")
        for field in ("id", "user_input", "reference"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"样本 {number}: 缺少非空字符串 {field}")
        if row["id"] in seen:
            raise ValueError(f"样本 ID 重复: {row['id']}")
        seen.add(row["id"])
        question = "".join(row["user_input"].split()).lower()
        if question in questions:
            raise ValueError(f"问题重复: {row['id']}")
        questions.add(question)
        for field in ("expected_domains", "reference_context_ids", "reference_contexts"):
            values = row.get(field)
            if not isinstance(values, list) or not values or any(
                not isinstance(v, str) or not v.strip() for v in values
            ):
                raise ValueError(f"样本 {row['id']}: {field} 必须为非空字符串列表")
        if set(row["expected_domains"]) - set(settings.domains.allowed):
            raise ValueError(f"样本 {row['id']}: expected_domains 含非法领域")
        ids = row["reference_context_ids"]
        if len(set(ids)) != len(ids) or len(ids) != len(row["reference_contexts"]):
            raise ValueError(f"样本 {row['id']}: 真值 chunk ID 重复或与正文数量不一致")
        if "reviewed" in row and not isinstance(row["reviewed"], bool):
            raise ValueError(f"样本 {row['id']}: reviewed 必须为布尔值")
    return rows


def load_dataset(path):
    path = Path(path)
    try:
        if path.suffix.lower() == ".json":
            rows = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(rows, dict):
                rows = rows.get("samples")
        else:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
                    if line.strip()]
    except json.JSONDecodeError as exc:
        raise ValueError(f"数据集 JSON 无效: {exc}") from exc
    return validate_rows(rows)


def load_source_chunks(index_path):
    from ingestion.source_store import SourceResolver
    resolver = SourceResolver()
    items, seen = [], set()
    with Path(index_path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                chunk_id = data.pop("chunk_id")
                if chunk_id in seen or data.get("domain") not in settings.domains.allowed:
                    raise ValueError("chunk ID 重复或 domain 非法")
                if {"content", "document", "title", "section_path"} & data.keys():
                    raise ValueError("必须使用 v5.0 无正文索引 JSONL")
                items.append(resolver.hydrate({"chunk_id": chunk_id, "metadata": data}))
                seen.add(chunk_id)
            except Exception as exc:
                raise ValueError(f"索引第 {line_number} 行不能回读: {exc}") from exc
    if not items:
        raise ValueError("索引中没有可用分块，请先运行 python ingest.py")
    return items


def verify_gold_in_index(rows, client):
    """逐个精确校验真值 ID、原文及版本；不做模糊匹配或预测回填真值。"""
    from utils.index_audit import assert_index_collection
    from pipeline.retrieval.retriever import get_available_domains
    from ingestion.source_store import SourceResolver
    wanted = sorted({cid for row in rows for cid in row["reference_context_ids"]})
    found = {}
    resolver = SourceResolver()
    for domain in get_available_domains(client):
        collection = client.get_collection(name=settings.paths.collection_prefix + domain,
                                           embedding_function=None)
        assert_index_collection(collection)
        for start in range(0, len(wanted), 128):
            result = collection.get(ids=wanted[start:start + 128], include=["metadatas"])
            for cid, metadata in zip(result["ids"], result["metadatas"]):
                if cid in found:
                    raise ValueError(f"chunk ID 出现在多个 Collection: {cid}")
                found[cid] = resolver.hydrate({"chunk_id": cid, "metadata": metadata})
    missing = set(wanted) - found.keys()
    if missing:
        raise ValueError(f"真值 chunk 不在当前向量库中（{len(missing)} 个）: {sorted(missing)[:5]}。"
                         "请核对 RAG_DATA_DIR，或入库后重新生成数据集。")
    for row in rows:
        versions = {s["chunk_id"]: s for s in row.get("reference_sources", [])}
        for cid, context in zip(row["reference_context_ids"], row["reference_contexts"]):
            if found[cid]["document"] != context:
                raise ValueError(f"样本 {row['id']}: {cid} 的真值正文与当前索引不一致")
            for key in ("source_ref", "content_hash"):
                if key in versions.get(cid, {}) and versions[cid][key] != found[cid]["metadata"].get(key):
                    raise ValueError(f"样本 {row['id']}: {cid} 的 {key} 已变更，请重新生成真值")
    return len(found)


def set_scores(predicted, expected):
    predicted, expected = set(predicted), set(expected)
    if not expected:
        raise ValueError("真值集合不能为空")
    tp, fp, fn = len(predicted & expected), len(predicted - expected), len(expected - predicted)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / len(predicted) if predicted else 0.0,
            "recall": tp / len(expected), "f1": 2 * tp / (2 * tp + fp + fn)}


def retrieval_metrics(row, result, k=0, *, router_enabled=None, selector_enabled=None):
    if k < 0:
        raise ValueError("retrieval_k 不能小于 0")
    if router_enabled is None:
        router_enabled = bool(settings.rag.router.enabled)
    if selector_enabled is None:
        selector_enabled = bool(settings.rag.context.selector.enabled)
    pool = result["candidate_pool"]
    pool = pool[:k] if k else pool
    recall = set_scores([item["chunk_id"] for item in pool], row["reference_context_ids"])
    metrics = {"retrieval_recall": recall["recall"]}
    if router_enabled:
        route = set_scores(result["router_domains"], row["expected_domains"])
        metrics.update({"router_f1": route["f1"], "router_counts": route})
    if selector_enabled:
        final = set_scores([item["chunk_id"] for item in result["selected_contexts"]],
                           row["reference_context_ids"])
        # This is a final-context exact ID diagnostic. It is not an LLM selector
        # score when the selector is disabled, so omit it in that mode.
        metrics["final_context_chunk_recall"] = final["recall"]
    return metrics


def checked_score(value):
    value = float(value)
    if not math.isfinite(value) or value < -1e-8 or value > 1 + 1e-8:
        raise ValueError(f"Ragas 返回无效分数: {value}")
    return min(1.0, max(0.0, value))



def _question_type_analysis(rows, metric_names):
    weights = {
        "single": 0.6,
        "same_domain_multi": 0.2,
        "cross_domain_multi": 0.2,
    }
    grouped = {}
    for row in rows:
        q = row.get("question_type", "unknown")
        grouped.setdefault(q, []).append(row)
    by_type = {}
    for q, items in grouped.items():
        by_type[q] = {"count": len(items), "metrics": {}}
        for m in metric_names:
            vals=[x[m] for x in items if x.get(m) is not None]
            by_type[q]["metrics"][m] = sum(vals)/len(vals) if vals else None
    weighted = {}
    for m in metric_names:
        total = 0.0
        used = 0.0
        for q, w in weights.items():
            if q in by_type and by_type[q]["metrics"].get(m) is not None:
                total += w * by_type[q]["metrics"][m]
                used += w
        weighted[m] = total / used if used else None
    return by_type, weighted

def summarize(rows, *, router_enabled=None, selector_enabled=None):
    if router_enabled is None:
        router_enabled = any("router_f1" in row for row in rows)
    if selector_enabled is None:
        selector_enabled = any("final_context_chunk_recall" in row for row in rows)
    summary = {"total": len(rows), "ok": sum(row["status"] == "ok" for row in rows),
               "failed": sum(row["status"] != "ok" for row in rows), "metrics": {}}
    metric_names = ["retrieval_recall", "context_recall", "answer_correctness"]
    if router_enabled:
        metric_names.insert(0, "router_f1")
    if selector_enabled:
        metric_names.append("final_context_chunk_recall")
    for name in metric_names:
        values = [row[name] for row in rows if row.get(name) is not None]
        summary["metrics"][name] = {
            "mean": sum(values) / len(values) if values else None,
            "valid": len(values), "missing": len(rows) - len(values),
        }
    if router_enabled:
        counts = [row["router_counts"] for row in rows if row.get("router_counts") is not None]
        tp, fp, fn = (sum(c[key] for c in counts) for key in ("tp", "fp", "fn"))
        summary["router_f1_micro"] = 2 * tp / (2 * tp + fp + fn) if counts else None
    by_type, weighted = _question_type_analysis(rows, metric_names)
    summary["by_question_type"] = by_type
    summary["weighted_overall"] = weighted
    return summary
