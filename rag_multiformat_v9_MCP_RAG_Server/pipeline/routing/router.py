# -*- coding: utf-8 -*-
"""
LLM Query Router

流程：
1. LLM 从用户问题提取技术关键词，仅用于查询分析/日志，并为后续 BM25/混合检索预留；
2. LLM Domain Router 只根据用户原始问题判断 topology / device / pcb / control / schematic 哪些领域；
3. 返回一个或多个领域，供 search.py 进入对应 Chroma Collection 检索。

关键词与领域路由解耦，避免上游关键词扩写影响 domain 判断。

Prompt 统一维护在 `configs/prompts.yaml`。
不再使用 Embedding threshold 做领域分类。
"""

from __future__ import annotations

from typing import List, TypedDict

from utils.config_loader import settings
from llm import BaseLLMClient, create_llm_client
from utils.config_loader import settings


class QueryAnalysis(TypedDict):
    keywords: List[str]
    domains: List[str]
    reason: str


def extract_keywords(
    query: str,
    llm: BaseLLMClient,
) -> List[str]:
    query = query.strip()
    if not query:
        raise ValueError("Query 不能为空。")

    messages = [
        {
            "role": "system",
            "content": settings.prompts.keyword_extract_system_prompt,
        },
        {
            "role": "user",
            "content": settings.prompts.keyword_extract_user_prompt_template.format(
                query=query,
            ),
        },
    ]

    try:
        data = llm.chat_json(messages)
    except Exception:
        # 关键词仅用于分析/日志，提取失败不应影响 Router 或向量检索。
        return []

    raw = data.get("keywords", []) if isinstance(data, dict) else []
    if not isinstance(raw, list):
        return []

    keywords: List[str] = []
    seen = set()

    for item in raw:
        text = str(item).strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            keywords.append(text)

    return keywords


def classify_domains(
    query: str,
    llm: BaseLLMClient,
) -> tuple[List[str], str]:
    domain_text = "\n".join(
        f"- {name}: {settings.domains.descriptions[name]}"
        for name in settings.domains.allowed
    )

    messages = [
        {
            "role": "system",
            "content": settings.prompts.domain_router_system_prompt,
        },
        {
            "role": "user",
            "content": settings.prompts.domain_router_user_prompt_template.format(
                query=query,
                domain_text=domain_text,
            ),
        },
    ]

    try:
        data = llm.chat_json(messages)
    except Exception:
        # Router 仅用于缩小检索范围，不应成为单点故障。
        # LLM 调用失败时降级为全局检索（返回空域列表），由 search 层触发全域 fallback。
        return [], "Router调用失败，触发全局检索。"

    if not isinstance(data, dict):
        return [], "LLM Router 返回结构不是 JSON object，触发全局检索。"

    raw_domains = data.get("domains", [])
    reason = str(data.get("reason", "")).strip()

    domains: List[str] = []
    if isinstance(raw_domains, list):
        for item in raw_domains:
            domain = str(item).strip().lower()
            if domain in settings.domains.allowed and domain not in domains:
                domains.append(domain)

    # Router 仅用于缩小检索范围，不应成为单点故障。
    # 当 LLM 无法判断领域时，返回空列表，由 search 层触发全域 fallback。
    if not domains:
        reason = reason or "Router未识别有效领域，触发全局检索。"

    return domains, reason


def analyze_query(
    query: str,
    llm: BaseLLMClient | None = None,
) -> QueryAnalysis:
    if not settings.rag.router.enabled:
        return {"keywords": [], "domains": [], "reason": "Router disabled by config"}
    if llm is None:
        llm = create_llm_client(role="router")

    keywords = extract_keywords(
        query=query,
        llm=llm,
    )

    domains, reason = classify_domains(
        query=query,
        llm=llm,
    )

    return {
        "keywords": keywords,
        "domains": domains,
        "reason": reason,
    }
