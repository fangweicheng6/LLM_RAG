
"""构造 RAG 真值评估集

默认分布:
- 60% single-hop: 单 chunk 可回答
- 20% same-domain multi-hop: 同一 domain 的多个 chunk 联合回答
- 20% cross-domain multi-hop: 不同 domain 的多个 chunk 联合回答

设计原则:
1. reference_context_ids 必须是真实存在的 chunk_id。
2. reference_contexts 与 reference_context_ids 一一对应。
3. 多跳问题必须至少使用两个证据片段。
4. evidence_quotes 必须来自对应原文连续片段。
5. 生成失败样本丢弃，不污染真值集。\n6. 默认增量生成，已有数据集时自动补齐；--rewrite强制重建。
"""

from __future__ import annotations

import argparse
import json
import random
import time
import shutil
from collections import defaultdict, Counter
from datetime import datetime, timezone
from pathlib import Path

from utils.config_loader import settings
from evaluation.eval_common import load_source_chunks, sha256_file, validate_rows, write_json, write_jsonl


DEFAULT_RATIO = settings.evaluation.dataset.ratio.as_dict()

# 问题意图分布：用于分析不同任务类型下的RAG能力
DEFAULT_INTENT_RATIO = {
    "knowledge_qa": 0.50,
    "reasoning": 0.30,
    "comparison": 0.10,
    "document_extract": 0.05,
    "table_analysis": 0.05,
}


def load_dataset_generation_config():
    """
    统一读取评估数据集配置。
    优先使用 configs/dataset_generation.yaml，
    避免 settings.py 与 yaml 双重控制导致数量不一致。
    """
    def _as_dict(value):
        # settings 里的嵌套配置是 ConfigNode，calculate_target_count 需要真正的 dict（调用 .items()）
        return value.as_dict() if hasattr(value, "as_dict") else value

    result = {
        "total_size": settings.get("dataset_generation.dataset.total_size", settings.evaluation.dataset.size),
        "question_type_ratio": _as_dict(settings.get("dataset_generation.dataset.question_type_ratio", DEFAULT_RATIO)),
        "question_intent_ratio": _as_dict(settings.get("dataset_generation.dataset.question_intent_ratio", DEFAULT_INTENT_RATIO)),
    }
    return result

def calculate_target_count(total_size, ratios):
    result = {k: int(total_size * v) for k, v in ratios.items()}
    diff = total_size - sum(result.values())
    result["single"] += diff
    return result


def calculate_ratio_count(total_size, ratios):
    """根据比例生成目标数量，最大余数分配给第一个类别。"""
    result = {k: int(total_size * v) for k, v in ratios.items()}
    diff = total_size - sum(result.values())
    if result:
        result[next(iter(result))] += diff
    return result


def normalize(s):
    return "".join(str(s).split()).lower()


def build_pools(items, seed=42):
    rng = random.Random(seed)

    by_domain = defaultdict(list)
    for x in items:
        if len(x["document"].strip()) >= 50:
            by_domain[x["metadata"]["domain"]].append(x)

    for v in by_domain.values():
        rng.shuffle(v)

    # 单跳同样过滤无效 chunk，避免 OCR 残片导致生成失败
    single = [x for x in items if len(x.get("document", "").strip()) >= 50]

    same = []
    for domain, chunks in by_domain.items():
        if len(chunks) >= 2:
            same.append(chunks)

    cross = []
    domains = list(by_domain)
    for i, d1 in enumerate(domains):
        for d2 in domains[i + 1:]:
            if by_domain[d1] and by_domain[d2]:
                cross.append((by_domain[d1], by_domain[d2]))

    return single, same, cross


def balanced_chunks(items, count, seed=42):
    """Legacy deterministic domain-balanced sampler retained for compatibility/tests.

    输入顺序不会影响结果；优先轮询各 domain，再从剩余 chunk 中补足。
    """
    if count <= 0 or not items:
        return []
    by_domain = {}
    for item in sorted(items, key=lambda x: (str((x.get("metadata") or {}).get("domain", "")), str(x.get("chunk_id", "")))):
        domain = str((item.get("metadata") or {}).get("domain", ""))
        by_domain.setdefault(domain, []).append(item)
    rng = random.Random(seed)
    domains = sorted(by_domain)
    for domain in domains:
        rng.shuffle(by_domain[domain])
    result = []
    positions = {d: 0 for d in domains}
    while len(result) < min(count, len(items)):
        progressed = False
        for domain in domains:
            i = positions[domain]
            if i < len(by_domain[domain]):
                result.append(by_domain[domain][i])
                positions[domain] += 1
                progressed = True
                if len(result) >= min(count, len(items)):
                    break
        if not progressed:
            break
    return result


def choose_pairs(pool, count, seed):
    """Legacy random sampler retained for compatibility/tests."""
    rng = random.Random(seed)
    result = []
    if not pool:
        return result
    for _ in range(count * 3):
        if len(result) >= count:
            break
        if isinstance(pool[0], tuple):
            a, b = rng.choice(pool)
            result.append([rng.choice(a), rng.choice(b)])
        else:
            arr = rng.choice(pool)
            if len(arr) >= 2:
                result.append(rng.sample(arr, 2))
    return result


def _balanced_candidates(pool, chunk_usage, chunk_attempts):
    """Return the least-used/least-attempted chunks in a pool."""
    if not pool:
        return []
    scored = []
    for chunk in pool:
        cid = chunk["chunk_id"]
        scored.append(((chunk_usage[cid], chunk_attempts[cid]), chunk))
    best_score = min(score for score, _ in scored)
    return [chunk for score, chunk in scored if score == best_score]


def choose_balanced_chunk(pool, chunk_usage, chunk_attempts, rng):
    """Prefer unseen chunks, then chunks with the lowest reuse count."""
    candidates = _balanced_candidates(pool, chunk_usage, chunk_attempts)
    return rng.choice(candidates) if candidates else None


def choose_balanced_same_domain(same_pool, chunk_usage, chunk_attempts, rng):
    """Choose two distinct low-usage chunks from the least-covered domain."""
    options = []
    for arr in same_pool:
        if len(arr) < 2:
            continue
        ordered = sorted(
            arr,
            key=lambda c: (
                chunk_usage[c["chunk_id"]],
                chunk_attempts[c["chunk_id"]],
                c["chunk_id"],
            ),
        )
        best2 = ordered[:2]
        score = (
            sum(chunk_usage[c["chunk_id"]] for c in best2),
            sum(chunk_attempts[c["chunk_id"]] for c in best2),
        )
        options.append((score, arr))
    if not options:
        return []
    best_score = min(score for score, _ in options)
    domain_arrays = [arr for score, arr in options if score == best_score]
    arr = rng.choice(domain_arrays)

    first = choose_balanced_chunk(arr, chunk_usage, chunk_attempts, rng)
    remaining = [c for c in arr if c["chunk_id"] != first["chunk_id"]]
    second = choose_balanced_chunk(remaining, chunk_usage, chunk_attempts, rng)
    return [first, second] if first and second else []


def choose_balanced_cross_domain(cross_pool, chunk_usage, chunk_attempts, rng):
    """Choose one low-usage chunk from each of two domains, preferring least-covered pairs."""
    options = []
    for a, b in cross_pool:
        ca = choose_balanced_chunk(a, chunk_usage, chunk_attempts, rng)
        cb = choose_balanced_chunk(b, chunk_usage, chunk_attempts, rng)
        if not ca or not cb:
            continue
        score = (
            chunk_usage[ca["chunk_id"]] + chunk_usage[cb["chunk_id"]],
            chunk_attempts[ca["chunk_id"]] + chunk_attempts[cb["chunk_id"]],
        )
        options.append((score, a, b))
    if not options:
        return []
    best_score = min(score for score, _, _ in options)
    a, b = rng.choice([(a, b) for score, a, b in options if score == best_score])
    ca = choose_balanced_chunk(a, chunk_usage, chunk_attempts, rng)
    cb = choose_balanced_chunk(b, chunk_usage, chunk_attempts, rng)
    return [ca, cb] if ca and cb else []


def ask_llm(client, prompt, retries=3):
    """带重试的评估集生成调用。避免第三方网关瞬断导致任务卡死。"""
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            print(f"    LLM request attempt {attempt}/{retries}", flush=True)
            result = client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=settings.evaluation.runtime.generation_max_tokens,
            )
            print("    LLM returned", flush=True)
            return result
        except Exception as e:
            last_error = e
            print(f"    LLM failed: {e}", flush=True)
            if attempt < retries:
                time.sleep(attempt * 2)
    raise RuntimeError(f"LLM generation failed after retries: {last_error}")


def generate_one(client, chunks, hop_type, target_intent=None):
    evidence = "\n\n".join(
        [f"[证据{i+1}] {c['document'][:2500]}" for i, c in enumerate(chunks)]
    )

    prompt = f"""
你是严格的工业RAG评估数据集构造器。

知识关系类型: {hop_type}\n目标问题意图: {target_intent or "不限"}\n\n请根据证据生成一个高质量中文问答，并同时标注问题属性。

要求：
1. 问题必须只能由给出的证据回答。
2. 不允许引入外部知识。
3. cross_domain_multi必须体现不同知识域之间的关联、影响或推理关系，禁止简单拼接事实。
4. single问题禁止生成跨域问题。
5. 禁止生成无关的日期、版权、表格复制类问题，除非intent明确为对应类型。

intent只能从以下选择：
- knowledge_qa: 知识问答
- reasoning: 因果/影响/设计推理
- comparison: 对比分析
- document_extract: 文档字段提取
- table_analysis: 表格数据分析

返回严格JSON：
{{
"question":"",
"answer":"",
"intent":"knowledge_qa",
"quotes":["原文连续片段1","原文连续片段2"]
}}

证据：
{evidence}
"""

    raw = ask_llm(client, prompt)
    text = raw.strip().replace("```json", "").replace("```", "")
    data = json.loads(text)

    if not data.get("question") or not data.get("answer"):
        raise ValueError("empty qa")

    allowed_intents={"knowledge_qa","reasoning","comparison","document_extract","table_analysis"}
    if data.get("intent") not in allowed_intents:
        raise ValueError("invalid intent")

    for q in data.get("quotes", []):
        if not any(q in c["document"] for c in chunks):
            raise ValueError("quote not found")

    return data


def llm_validate_dataset(client, rows, batch_size=25):
    """最终数据集自检：分批检查全部样本，而不是只检查前30条。"""
    all_valid = []
    all_invalid = []
    reasons = {}
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        prompt=f"""
你是RAG评估集质量审核器。
检查下面数据。
逐条判断：
1. question是否能由reference回答
2. answer是否回答question
3. question_type是否合理
4. question_intent是否合理
5. gold chunks是否足够支撑答案

只返回JSON：
{{"valid_ids":[],"invalid_ids":[],"reason":{{}}}}

数据：
{json.dumps(batch,ensure_ascii=False)}
"""
        raw=ask_llm(client,prompt)
        text=raw.strip().replace("```json","").replace("```","")
        result=json.loads(text)
        all_valid.extend(result.get("valid_ids", []))
        all_invalid.extend(result.get("invalid_ids", []))
        reasons.update(result.get("reason", {}))
    return {"valid_ids": all_valid, "invalid_ids": all_invalid, "reason": reasons}


def next_sample_id(rows):
    """返回永久递增ID，避免增量生成覆盖旧样本。"""
    max_id = 0
    for row in rows:
        try:
            max_id = max(max_id, int(str(row.get("id", "")).split("_")[-1]))
        except Exception:
            continue
    return max_id + 1


def validate_unique_ids(rows):
    """保存前检查样本ID唯一性。"""
    ids = [r.get("id") for r in rows]
    dup = {x for x in ids if ids.count(x) > 1}
    if dup:
        raise RuntimeError(f"样本ID重复: {sorted(list(dup))}")


def normalize_dataset_ids(rows):
    """
    兼容旧版本错误生成的数据。
    不重排正常ID，只删除重复项，保留首次出现记录。
    """
    result = []
    seen = set()
    for row in rows:
        rid = row.get("id")
        if rid in seen:
            continue
        seen.add(rid)
        result.append(row)
    return result

def generate(items, size, output, seed, index_path, ratio=None, intent_ratio=None, rewrite=False):
    from evaluation.ragas_adapter import ProjectRagasLLM
    client = ProjectRagasLLM().client

    target = calculate_target_count(size, ratio or DEFAULT_RATIO)
    intent_target = calculate_ratio_count(size, intent_ratio or DEFAULT_INTENT_RATIO)
    intent_counts = {k: 0 for k in intent_target}
    single_pool, same_pool, cross_pool = build_pools(items, seed)

    pools = {
        "single": single_pool,
        "same_domain_multi": same_pool,
        "cross_domain_multi": cross_pool,
    }

    accepted = []
    rejected = []
    seen = set()
    counts = {k: 0 for k in target}
    by_id = {x["chunk_id"]: x for x in items}
    rng = random.Random(seed)
    # Coverage-aware sampling state. Usage is counted only for accepted rows;
    # attempts prevents repeatedly retrying a low-quality chunk that keeps failing generation.
    chunk_usage = Counter()
    chunk_attempts = Counter()

    # v7.10: 默认增量生成，已有评估集则在原数量基础上补齐。
    # 使用 --rewrite 才会清空重新生成。
    output = Path(output)
    if output.exists() and not rewrite:
        try:
            old_rows = [x for x in (json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()) if x]
            if old_rows:
                accepted.extend(old_rows[:size])
                accepted = normalize_dataset_ids(accepted)
                validate_unique_ids(accepted)
                for row in accepted:
                    counts[row.get("question_type", "single")] = counts.get(row.get("question_type", "single"), 0) + 1
                    intent_counts[row.get("question_intent", "knowledge_qa")] = intent_counts.get(row.get("question_intent", "knowledge_qa"), 0) + 1
                    seen.add(normalize(row.get("user_input", "")))
                    for cid in row.get("reference_context_ids", []):
                        chunk_usage[cid] += 1
                print(f"Resume mode: loaded existing dataset {len(accepted)}/{size}")
        except Exception as e:
            print(f"Resume load failed, start fresh: {e}")

    if rewrite:
        print("Rewrite mode enabled: ignore existing dataset.")

    attempts = 0
    # v7.9.1: 增加生成容错次数，避免最后少量cross-domain样本卡死
    max_attempts = size * settings.evaluation.dataset.generation_attempts_factor

    def generate_one_sample():
        """生成并校验一条样本；成功则加入 accepted 并更新计数，失败返回 None。"""
        nonlocal attempts
        attempts += 1

        remain_types = [k for k, v in target.items() if counts[k] < v]
        if not remain_types:
            return None
        hop_type = rng.choice(remain_types)

        remain_intents = [k for k, v in intent_target.items() if intent_counts[k] < v]
        target_intent = rng.choice(remain_intents) if remain_intents else "knowledge_qa"

        if hop_type == "single":
            chunk = choose_balanced_chunk(single_pool, chunk_usage, chunk_attempts, rng)
            if not chunk:
                return None
            chunks = [chunk]
        elif hop_type == "same_domain_multi":
            chunks = choose_balanced_same_domain(same_pool, chunk_usage, chunk_attempts, rng)
            if not chunks:
                return None
        else:
            chunks = choose_balanced_cross_domain(cross_pool, chunk_usage, chunk_attempts, rng)
            if not chunks:
                return None

        print(f"Attempt {attempts} | accepted {len(accepted)}/{size} | type={hop_type} | intent={target_intent} |  type_count={counts[hop_type]}/{target[hop_type]}", flush=True)

        try:
            qa = generate_one(client, chunks, hop_type, target_intent)
            validation_flags = {
                "json_format": True,
                "intent_match": qa.get("intent") == target_intent,
                "duplicate_check": normalize(qa["question"]) not in seen,
                "quote_check": True,
            }

            # 实时输出每条样本质量检查结果
            print("Validation:", flush=True)
            for check_name, check_value in validation_flags.items():
                print(
                    f"  [{'PASS' if check_value else 'FAIL'}] {check_name}",
                    flush=True
                )

            if not validation_flags["intent_match"]:
                raise ValueError(f"intent mismatch: expected={target_intent}, got={qa.get('intent')}")
            if not validation_flags["duplicate_check"]:
                raise ValueError("duplicate question")

            ids = [c["chunk_id"] for c in chunks]
            row = {
                "id": f"eval_{next_sample_id(accepted):05d}",
                "user_input": qa["question"],
                "reference": qa["answer"],
                "reference_context_ids": ids,
                "reference_contexts": [c["document"] for c in chunks],
                "expected_domains": sorted(list({c["metadata"]["domain"] for c in chunks})),
                "reference_sources": [
                    {"chunk_id": c["chunk_id"], "domain": c["metadata"]["domain"], "source_file": c["metadata"].get("source_file"), "relative_path": c["metadata"].get("relative_path")}
                    for c in chunks
                ],
                "evidence_quotes": [{"chunk_id": c["chunk_id"], "text": q} for c, q in zip(chunks, qa["quotes"])],
                "question_type": hop_type,
                "question_intent": target_intent,
                "difficulty": ("easy" if hop_type == "single" else ("medium" if hop_type == "same_domain_multi" else "hard")),
                "hard_negative_context_ids": [],
                "label_origin": "controlled_synthesis",
                "reviewed": False,
            }
            accepted.append(row)
            print(
                f"Accepted: {row['id']} | "
                f"type={hop_type} | intent={target_intent}",
                flush=True
            )
            counts[hop_type] += 1
            intent_counts[row["question_intent"]] = intent_counts.get(row["question_intent"], 0) + 1
            seen.add(normalize(row["user_input"]))
            for cid in ids:
                chunk_usage[cid] += 1
            return row
        except Exception as e:
            for c in chunks:
                chunk_attempts[c["chunk_id"]] += 1
            print(
                f"Validation FAILED: {str(e)} -> discarded",
                flush=True
            )
            rejected.append({
                "type": hop_type,
                "intent": target_intent,
                "error": str(e)
            })
            return None

    while len(accepted) < size and attempts < max_attempts:
        generate_one_sample()

    # v7.9.1: 在最终质量复核前保存一次中间结果，防止复核阶段导致全部丢失
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output.with_name(output.stem + "_partial.jsonl"), accepted)

    # 最终LLM质量自检 + 补齐：不合格样本被移除后，回退计数并继续生成补足，
    # 直到达到目标数量或超出尝试上限。只复核新增样本，避免重复复核已通过的样本。
    validated_ids = set()  # 已通过最终复核的样本 id
    while True:
        new_rows = [x for x in accepted if x["id"] not in validated_ids]
        if not new_rows:
            break
        try:
            checked = llm_validate_dataset(client, new_rows)
        except Exception as e:
            rejected.append({"type": "dataset_validation", "error": str(e)})
            break
        invalid = set(checked.get("invalid_ids", []))
        for x in new_rows:
            if x["id"] not in invalid:
                validated_ids.add(x["id"])
        if not invalid:
            break

        removed = {x["id"]: x for x in new_rows if x["id"] in invalid}
        accepted = [x for x in accepted if x["id"] not in removed]
        for row in removed.values():
            counts[row.get("question_type", "single")] = max(0, counts.get(row.get("question_type", "single"), 0) - 1)
            intent_counts[row.get("question_intent", "knowledge_qa")] = max(0, intent_counts.get(row.get("question_intent", "knowledge_qa"), 0) - 1)
            seen.discard(normalize(row.get("user_input", "")))
            for cid in row.get("reference_context_ids", []):
                chunk_usage[cid] = max(0, chunk_usage[cid] - 1)
        print(f"Final validation removed {len(removed)} sample(s); backfilling...", flush=True)
        if attempts >= max_attempts:
            break
        while len(accepted) < size and attempts < max_attempts:
            generate_one_sample()

    output=Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # 保存前最后一道数据完整性检查
    accepted = normalize_dataset_ids(accepted)
    validate_unique_ids(accepted)

    # 自动备份旧版本
    if output.exists():
        backup_dir = output.parent / "eval_dataset_backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / f"{output.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        shutil.copy2(output, backup)

    # 按ID排序，保证文件稳定
    accepted.sort(key=lambda x: int(str(x.get("id","eval_0")).split("_")[-1]))

    write_jsonl(output,accepted)
    write_json(output.with_suffix(".meta.json"),{
        "schema_version":3,
        "requested":size,
        "generated":len(accepted),
        "target":target,
        "counts":counts,
        "intent_target":intent_target,
        "intent_counts":intent_counts,
        "distribution":ratio or DEFAULT_RATIO,
        "intent_distribution":intent_ratio or DEFAULT_INTENT_RATIO,
        "attempts":attempts,
        "created_at":datetime.now(timezone.utc).isoformat(),
        "index_sha256":sha256_file(index_path),
        "rejected":rejected,
    })
    write_jsonl(output.parent/"failed_generation.jsonl", rejected)

    # 生成质量统计报告：同时报告“唯一chunk覆盖”和“引用槽位重复率”。
    final_usage = Counter(cid for x in accepted for cid in x.get("reference_context_ids", []))
    eligible_ids = {x["chunk_id"] for x in single_pool}
    unique_gold_chunks = len(final_usage)
    total_gold_references = sum(final_usage.values())
    per_domain_unique = Counter()
    for cid in final_usage:
        item = by_id.get(cid)
        if item:
            per_domain_unique[item["metadata"]["domain"]] += 1
    report = {
        "requested": size,
        "accepted": len(accepted),
        "attempts": attempts,
        "question_type_counts": counts,
        "question_intent_counts": intent_counts,
        "intent_target": intent_target,
        "failure_count": len(rejected),
        "eligible_chunks": len(eligible_ids),
        "unique_gold_chunks": unique_gold_chunks,
        "total_gold_references": total_gold_references,
        "gold_chunk_coverage_ratio": (unique_gold_chunks / len(eligible_ids)) if eligible_ids else 0.0,
        "reference_uniqueness_ratio": (unique_gold_chunks / total_gold_references) if total_gold_references else 0.0,
        "max_chunk_reuse": max(final_usage.values(), default=0),
        "reused_chunk_count": sum(1 for n in final_usage.values() if n > 1),
        "per_domain_unique_gold_chunks": dict(per_domain_unique),
        "top_reused_chunks": final_usage.most_common(20),
    }
    write_json(output.parent/"dataset_report.json", report)

    print("=== generation finished ===", flush=True)
    print(f"attempts={attempts}", flush=True)
    print(f"accepted={len(accepted)}/{size}", flush=True)
    print(f"question_type_counts={counts}", flush=True)
    print(f"question_intent_counts={intent_counts}", flush=True)
    print(f"unique_gold_chunks={report['unique_gold_chunks']}/{report['eligible_chunks']} "
          f"({report['gold_chunk_coverage_ratio']:.1%} corpus coverage)", flush=True)
    print(f"total_gold_references={report['total_gold_references']}; "
          f"reference_uniqueness={report['reference_uniqueness_ratio']:.1%}; "
          f"max_chunk_reuse={report['max_chunk_reuse']}", flush=True)

    # v7.9.1: 工业评测数据生成不因极少量样本失败而中断
    # 保存当前高质量结果，并输出缺失类别信息
    if len(accepted) != size:
        missing_type = {
            k: max(0, target[k] - counts.get(k, 0))
            for k in target
        }
        print(
            f"Warning: dataset generation incomplete: {len(accepted)}/{size}; "
            f"missing_type={missing_type}. Partial dataset has been saved.",
            flush=True
        )


def load_benchmark_view_config():
    """读取benchmark视图生成开关"""
    cfg = Path(__file__).parent / "configs" / "dataset_generation.yaml"
    result = {
        "router": True,
        "retrieval": True,
        "generation": True,
        "e2e": True,
    }
    try:
        for line in cfg.read_text(encoding="utf-8").splitlines():
            line=line.strip()
            if line.startswith("router:"):
                result["router"] = line.split(":")[-1].strip().lower()=="true"
            elif line.startswith("retrieval:"):
                result["retrieval"] = line.split(":")[-1].strip().lower()=="true"
            elif line.startswith("generation:"):
                result["generation"] = line.split(":")[-1].strip().lower()=="true"
            elif line.startswith("e2e:"):
                result["e2e"] = line.split(":")[-1].strip().lower()=="true"
    except Exception:
        pass
    return result


def build_benchmark_views(dataset_file):
    """生成Benchmark视图，无需用户手动执行"""
    try:
        from evaluation.benchmark_splitter import split_benchmarks
        switches = load_benchmark_view_config()
        output_dir = Path(dataset_file).parent / "benchmarks"
        split_benchmarks(
            dataset_file,
            output_dir,
            enabled=switches
        )
        print(f"benchmark views generated: {output_dir}")
    except Exception as e:
        print(f"benchmark view generation skipped: {e}")


def main():
    print("=== eval dataset start ===", flush=True)
    parser = argparse.ArgumentParser()
    dataset_cfg = load_dataset_generation_config()
    parser.add_argument(
        "--size",
        type=int,
        default=dataset_cfg["total_size"],
        help="覆盖配置文件中的dataset.total_size"
    )
    parser.add_argument("--output", default=str(settings.evaluation.paths.dataset))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--index", default=str(settings.paths.chunk_jsonl))
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="忽略已有eval_dataset，重新生成"
    )
    args = parser.parse_args()

    print("loading chunks...", flush=True)
    items = load_source_chunks(args.index)
    print("generating questions...", flush=True)
    print(f"target size={args.size}; output={args.output}", flush=True)
    generate(
        items,
        args.size,
        args.output,
        args.seed,
        args.index,
        dataset_cfg["question_type_ratio"],
        dataset_cfg["question_intent_ratio"],
        args.rewrite,
    )
    build_benchmark_views(args.output)
    

if __name__ == "__main__":
    main()
