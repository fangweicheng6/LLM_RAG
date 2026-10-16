"""将统一评估集拆分为 Router / Retrieval / Generation / E2E 四类 benchmark。"""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path

from utils.config_loader import settings


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _load_rows(input_file: Path) -> list[dict]:
    from evaluation.eval_common import load_dataset
    return load_dataset(input_file)


def split_benchmarks(input_file, output_dir, enabled=None):
    """拆分并返回 manifest。

    Router: 只保留路由真值；
    Retrieval: 只保留检索真值，不要求生成答案；
    Generation: 保留 Gold Context，绕过检索直接评估生成；
    E2E: 保留完整样本，默认仅纳入知识问答/推理/比较意图。
    """
    if enabled is None:
        enabled = {"router": True, "retrieval": True, "generation": True, "e2e": True}

    input_file = Path(input_file).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(input_file)

    router, retrieval, generation, e2e = [], [], [], []
    common_meta = ("question_type", "question_intent", "difficulty", "reviewed", "label_origin")

    for r in rows:
        meta = {key: r[key] for key in common_meta if key in r}
        router.append({
            "id": r["id"],
            "user_input": r["user_input"],
            "expected_domains": r.get("expected_domains", []),
            **meta,
        })
        retrieval.append({
            "id": r["id"],
            "user_input": r["user_input"],
            "expected_domains": r.get("expected_domains", []),
            "reference_context_ids": r.get("reference_context_ids", []),
            **meta,
        })
        generation.append({
            "id": r["id"],
            "user_input": r["user_input"],
            "reference": r["reference"],
            "reference_context_ids": r.get("reference_context_ids", []),
            "reference_contexts": r.get("reference_contexts", []),
            "reference_sources": r.get("reference_sources", []),
            **meta,
        })
        # if r.get("question_intent") in set(settings.evaluation.benchmark.e2e_intents):
        e2e.append(r)

    datasets = {
        "router": ("router_test.jsonl", router),
        "retrieval": ("retrieval_test.jsonl", retrieval),
        "generation": ("generation_test.jsonl", generation),
        "e2e": ("e2e_test.jsonl", e2e),
    }
    counts = {}
    for key, (name, data) in datasets.items():
        counts[key] = len(data)
        if enabled.get(key, True):
            _write_jsonl(out / name, data)

    project_root = Path(__file__).resolve().parent
    try:
        source_display = str(input_file.relative_to(project_root))
    except ValueError:
        source_display = str(input_file)

    manifest = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": source_display,
        "source_sha256": _sha256(input_file),
        "source_count": len(rows),
        "e2e_intents": list(settings.evaluation.benchmark.e2e_intents),
        "counts": counts,
        "files": {key: name for key, (name, _) in datasets.items()},
    }
    (out / "benchmark_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="拆分四类 RAG benchmark 数据集")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    info = split_benchmarks(a.input, a.output)
    print(json.dumps(info, ensure_ascii=False, indent=2))
