# -*- coding: utf-8 -*-
"""
Grounded Answer Generator

使用 Context Selector 选出的上下文生成最终回答。
与具体 LLM Provider 解耦，只依赖统一 BaseLLMClient。
Prompt 统一维护在 `configs/prompts.yaml`。
"""

from __future__ import annotations

from typing import Any, Dict, List

from pipeline.generation.guard import sanitize_answer
from utils.config_loader import settings
from llm import BaseLLMClient
from ingestion.source_store import format_location
from observability import langfuse_span, update_span


def _build_grounded_context(
    selected_contexts: List[Dict[str, Any]],
) -> tuple[str, List[Dict[str, str]]]:
    parts: List[str] = []
    sources: List[Dict[str, str]] = []

    for index, item in enumerate(selected_contexts, start=1):
        meta = item.get("metadata") or {}
        label = f"C{index}"
        chunk_id = str(item.get("chunk_id", ""))
        source_file = str(meta.get("source_file", ""))
        section = str(meta.get("section_path", ""))
        location = format_location(meta)

        parts.append(
            f"[{label}]\n"
            f"chunk_id: {chunk_id}\n"
            f"source: {source_file}\n"
            f"section: {section}\n"
            f"location: {location}\n"
            f"content:\n{item.get('document', '')}\n"
        )

        sources.append(
            {
                "label": label,
                "chunk_id": chunk_id,
                "source_file": source_file,
                "section": section,
                "location": location,
            }
        )

    return "\n".join(parts), sources


def generate_answer(
    *,
    query: str,
    selected_contexts: List[Dict[str, Any]],
    llm: BaseLLMClient,
    context_sufficient: bool = True,
    missing_information: List[str] | None = None,
) -> tuple[str, List[Dict[str, str]]]:
    """根据已选择上下文生成最终、可追溯的回答。"""
    if not selected_contexts:
        return (
            "当前知识库候选结果中没有足够的相关上下文，无法基于知识库可靠回答该问题。",
            [],
        )

    context_text, sources = _build_grounded_context(selected_contexts)
    missing_information = missing_information or []

    # 确定性拦截：Selector 已判定上下文不充分时，不调用 LLM 凭空生成。
    # 否则模型可能在缺少依据时编造具体数值/公式/结论，违背“内容必须 KB-based”的验收目标。
    # 这里明确返回“无法可靠回答”，并附上缺失点与已找到的出处，供上层与评测记录溯源。
    if not context_sufficient:
        missing_text = "；".join(missing_information) if missing_information else "无"
        source_lines = "\n".join(
            f"- {s['label']}：{s['source_file']}"
            + (f"（{s['section']}）" if s.get("section") else "")
            + (f"（{s['location']}）" if s.get("location") else "")
            for s in sources
        )
        refusal = (
            "当前知识库中的上下文不足以可靠回答该问题，为避免编造，不生成答案。\n"
            f"缺失信息：{missing_text}\n"
            f"已找到的相关出处：\n{source_lines}"
        )
        return refusal, sources

    messages = [
        {
            "role": "system",
            "content": settings.prompts.answer_system_prompt,
        },
        {
            "role": "user",
            "content": settings.prompts.answer_user_prompt_template.format(
                query=query,
                context_sufficient=context_sufficient,
                missing_information=(
                    "; ".join(missing_information)
                    if missing_information
                    else "无"
                ),
                context_text=context_text,
            ),
        },
    ]

    with langfuse_span("generation", input={"query": query, "context_count": len(selected_contexts)}) as generation_span:
        raw_answer = llm.chat(
            messages,
            max_tokens=settings.model.generation.max_output_tokens,
        )
        usage = getattr(raw_answer, "usage", None)
        update_span(
            generation_span,
            {
                "answer_length": len(raw_answer),
                "context_sources": [s.get("chunk_id", "") for s in sources],
                "token_usage": usage if isinstance(usage, dict) else None,
            },
        )

    # Provider 返回的是一次性完整文本；这里做最后一道确定性质量保护：
    # 去掉模型自己生成的来源附录，并截断明显的尾部循环重复。
    answer = sanitize_answer(raw_answer)

    return answer, sources
