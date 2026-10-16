# -*- coding: utf-8 -*-
"""
LLM Context Selector

向量检索只负责生成候选结果池。
LLM 负责：
1. 选择“最小充分上下文”；
2. 判断当前上下文是否充分；
3. 若不充分，明确提出结构化补全请求。

补全策略优先级：
same_section -> parent_section -> targeted_search -> neighbors
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from utils.config_loader import settings
from llm import BaseLLMClient
from ingestion.source_store import format_location


class ExpansionRequest(TypedDict, total=False):
    type: str
    anchor_chunk_id: str
    query: str
    domain: str
    before: int
    after: int
    reason: str


class ContextSelection(TypedDict):
    selected_chunk_ids: List[str]
    sufficient: bool
    missing_information: List[str]
    expansion_requests: List[ExpansionRequest]
    reason: str


def _candidate_text(item: Dict[str, Any], index: int) -> str:
    meta = item.get("metadata") or {}

    location = format_location(meta)
    section_index = meta.get("section_chunk_index")
    section_count = meta.get("section_chunk_count")
    section_part = ""
    if section_index is not None and section_count is not None:
        section_part = f"{int(section_index) + 1}/{int(section_count)}"

    similarity = item.get("similarity")
    similarity_text = (
        f"{float(similarity):.4f}"
        if isinstance(similarity, (int, float))
        else "n/a"
    )

    return (
        f"### Candidate {index}\n"
        f"chunk_id: {item.get('chunk_id', '')}\n"
        f"domain: {meta.get('domain', '')}\n"
        f"source: {meta.get('source_file', '')}\n"
        f"section: {meta.get('section_path', '')}\n"
        f"section_id: {meta.get('section_id', '')}\n"
        f"section_part: {section_part}\n"
        f"location: {location}\n"
        f"retrieval_source: {item.get('retrieval_source', 'vector')}\n"
        f"vector_similarity: {similarity_text}\n"
        f"content:\n{item.get('document', '')}\n"
    )


def build_candidate_prompt_text(
    candidates: List[Dict[str, Any]],
) -> tuple[str, List[Dict[str, Any]]]:
    """受字符预算保护地组装候选池，不代表最终 Top-K。"""
    selected_for_prompt: List[Dict[str, Any]] = []
    parts: List[str] = []
    current_chars = 0

    for index, item in enumerate(candidates, start=1):
        text = _candidate_text(item, index)
        if parts and current_chars + len(text) > settings.rag.retrieval.candidate_pool_max_chars:
            break

        parts.append(text)
        selected_for_prompt.append(item)
        current_chars += len(text)

    return "\n".join(parts), selected_for_prompt


def _normalize_expansion_requests(
    raw_requests: Any,
    *,
    valid_ids: set[str],
) -> List[ExpansionRequest]:
    if not isinstance(raw_requests, list):
        return []

    allowed_types = {
        "same_section",
        "parent_section",
        "targeted_search",
        "neighbors",
    }
    result: List[ExpansionRequest] = []

    for raw in raw_requests:
        if not isinstance(raw, dict):
            continue

        request_type = str(raw.get("type", "")).strip()
        if request_type not in allowed_types:
            continue

        request: ExpansionRequest = {"type": request_type}
        anchor = str(raw.get("anchor_chunk_id", "")).strip()

        if request_type in {"same_section", "parent_section", "neighbors"}:
            if anchor not in valid_ids:
                continue
            request["anchor_chunk_id"] = anchor

        if request_type == "targeted_search":
            query = str(raw.get("query", "")).strip()
            if not query:
                continue
            request["query"] = query
            domain = str(raw.get("domain", "")).strip().lower()
            if domain:
                request["domain"] = domain
            if anchor in valid_ids:
                request["anchor_chunk_id"] = anchor

        if request_type == "neighbors":
            for key in ("before", "after"):
                try:
                    value = int(raw.get(key, 1))
                except Exception:
                    value = 1
                request[key] = max(0, min(value, 3))

        reason = str(raw.get("reason", "")).strip()
        if reason:
            request["reason"] = reason

        result.append(request)

    return result


def _normalize_bool(value: Any) -> bool:
    """把 LLM 可能返回的字符串布尔值归一化为真布尔。

    部分 OpenAI 兼容模型即使提示要求 JSON boolean，仍可能输出字符串
    "false"/"no"/"0"。bool("false") 会因非空字符串而误判为 True，
    导致“不充分”被当成“充分”，静默跳过自我完善。这里显式归一化。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return False


def _same_section_fallback(
    selected_contexts: List[Dict[str, Any]],
    reason: str,
) -> List[ExpansionRequest]:
    """确定性兜底：当 LLM 判定“不充分”却未给出可执行请求（或解析失败）时，
    若已选 chunk 明确属于一个被拆成多段的章节，则自动补 same_section。

    只做最安全的第一优先级补全，不自动猜 parent/targeted/neighbors。
    """
    for item in selected_contexts:
        meta = item.get("metadata") or {}
        try:
            section_count = int(meta.get("section_chunk_count", 1))
        except Exception:
            section_count = 1
        if section_count > 1:
            return [{
                "type": "same_section",
                "anchor_chunk_id": str(item.get("chunk_id", "")),
                "reason": reason,
            }]
    return []


def select_contexts(
    *,
    query: str,
    candidates: List[Dict[str, Any]],
    llm: BaseLLMClient,
    allow_expansion: bool = True,
) -> tuple[ContextSelection, List[Dict[str, Any]]]:
    if not candidates:
        return {
            "selected_chunk_ids": [],
            "sufficient": False,
            "missing_information": ["候选结果池为空"],
            "expansion_requests": [],
            "reason": "候选结果池为空。",
        }, []

    candidate_text, prompt_candidates = build_candidate_prompt_text(candidates)
    valid_ids = {
        str(item.get("chunk_id", ""))
        for item in prompt_candidates
        if item.get("chunk_id")
    }

    expansion_rule = (
        settings.prompts.context_selector_expansion_allowed_prompt
        if allow_expansion
        else settings.prompts.context_selector_expansion_disabled_prompt
    )

    messages = [
        {
            "role": "system",
            "content": (
                settings.prompts.context_selector_system_prompt
                + expansion_rule
                + "只返回 JSON。"
            ),
        },
        {
            "role": "user",
            "content": settings.prompts.context_selector_user_prompt_template.format(
                query=query,
                candidate_text=candidate_text,
            ),
        },
    ]

    # Selector JSON 稳定性保护：
    # 部分 OpenAI-compatible 模型即使要求 JSON，仍可能返回解释文本。
    # 这里增加专用重试，而不是让整个 E2E pipeline 失败。
    selector_retry = 3
    data = None
    last_error = None

    retry_messages = messages
    for attempt in range(selector_retry):
        try:
            data = llm.chat_json(retry_messages, max_tokens=1024)
            if isinstance(data, dict):
                break
            raise RuntimeError("Selector 返回非 JSON object。")
        except Exception as exc:
            last_error = exc
            retry_messages = [
                {
                    "role": "system",
                    "content": (
                        "你是严格 JSON API。\n"
                        "禁止输出任何解释、Markdown、代码块。\n"
                        "只能输出 JSON object。\n"
                        "必须包含字段 selected_chunk_ids。\n"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "重新执行上下文选择。\n"
                        + messages[-1]["content"]
                    ),
                },
            ]

    if not isinstance(data, dict):
        # 最终安全降级：不阻断整个 RAG 流程。
        # 关键：JSON 解析失败意味着“无法判断充分性”，必须保守地按“不充分”处理——
        # 否则自我完善循环（not sufficient and expansion_requests）永远不触发，
        # 生成器还会被谎称上下文已充分。这里按不充分处理，并走确定性 same_section 兜底。
        selected_contexts = prompt_candidates[:3]
        selected_ids = [
            str(item.get("chunk_id"))
            for item in selected_contexts
            if item.get("chunk_id")
        ]
        expansion_requests = (
            _same_section_fallback(
                selected_contexts,
                "Selector JSON解析失败，系统保守按不充分处理并自动补齐同章节。",
            )
            if allow_expansion
            else []
        )
        return {
            "selected_chunk_ids": selected_ids,
            "sufficient": False,
            "missing_information": ["Selector JSON解析失败，无法判断上下文充分性。"],
            "expansion_requests": expansion_requests,
            "reason": (
                "Selector JSON解析失败，自动回退Top3候选并按不充分处理。"
                + (f" error={last_error}" if last_error else "")
            ),
        }, selected_contexts

    raw_ids = data.get("selected_chunk_ids", [])
    reason = str(data.get("reason", "")).strip()
    sufficient = _normalize_bool(data.get("sufficient", False))

    raw_missing = data.get("missing_information", [])
    missing_information = [
        str(x).strip()
        for x in raw_missing
        if str(x).strip()
    ] if isinstance(raw_missing, list) else []

    selected_ids: List[str] = []
    if isinstance(raw_ids, list):
        for item in raw_ids:
            chunk_id = str(item).strip()
            if chunk_id in valid_ids and chunk_id not in selected_ids:
                selected_ids.append(chunk_id)

    by_id = {
        str(item.get("chunk_id")): item
        for item in prompt_candidates
        if item.get("chunk_id")
    }
    selected_contexts = [
        by_id[chunk_id]
        for chunk_id in selected_ids
        if chunk_id in by_id
    ]

    # Selector 容错：LLM 可能返回空选择。
    # 不允许把有效候选全部丢弃，保留 reranker 前若干结果作为安全上下文。
    if not selected_contexts:
        selected_contexts = prompt_candidates[:3]
        selected_ids = [
            str(item.get("chunk_id"))
            for item in selected_contexts
            if item.get("chunk_id")
        ]
        reason = (reason + "；Selector为空，启用Top3候选保护。").strip("；")

    expansion_requests = (
        _normalize_expansion_requests(
            data.get("expansion_requests", []),
            valid_ids=valid_ids,
        )
        if allow_expansion and not sufficient
        else []
    )

    # 确定性兜底：如果 LLM 已明确判断“不充分”，却没有给出可执行请求，
    # 且已选 Chunk 明确属于一个被拆成多段的章节，则自动补 same_section。
    # 这里只做最安全的第一优先级补全，不自动猜 parent/targeted/neighbors。
    if allow_expansion and not sufficient and not expansion_requests:
        expansion_requests = _same_section_fallback(
            selected_contexts,
            "系统检测到该章节被拆成多个 Chunk，自动补齐同章节。",
        )

    return {
        "selected_chunk_ids": selected_ids,
        "sufficient": sufficient,
        "missing_information": missing_information,
        "expansion_requests": expansion_requests,
        "reason": reason,
    }, selected_contexts
