# -*- coding: utf-8 -*-
"""Multi-hop query decomposition with cross-domain planning."""
from __future__ import annotations

from typing import List
from utils.config_loader import settings


def expand_query_to_subqueries(query: str, llm, domains: List[str] | None = None) -> List[str]:
    """
    将复杂问题拆成多跳检索query。
    v7.1:
    - 保留原始query
    - 支持跨领域多跳
    - 根据Router domains生成领域视角query
    """
    if not settings.rag.retrieval.multi_hop.enabled:
        return [query]

    domains = domains or []

    domain_hint = ""
    if settings.rag.retrieval.multi_hop.cross_domain_enabled and domains:
        domain_hint = (
            "\n当前Router识别领域："
            + ", ".join(domains)
            + "\n请针对不同领域生成必要的检索子问题。"
        )

    prompt = [
        {
            "role": "system",
            "content": (
                "你是电力电子RAG检索规划器。"
                "对于复杂问题，请拆分成最多3个互补检索子问题。"
                "如果问题涉及多个知识领域，请分别覆盖不同领域的信息。"
                "子问题必须用于检索文档，不要回答问题。"
                "只返回JSON: {\"queries\":[]}。"
            ),
        },
        {
            "role": "user",
            "content": query + domain_hint,
        },
    ]

    try:
        data = llm.chat_json(prompt)
        qs = data.get("queries", []) if isinstance(data, dict) else []

        result = [query]
        for q in qs:
            q = str(q).strip()
            if q and q not in result:
                result.append(q)

        # result 首元素固定为原始 query，其余才是「子问题」。max_subqueries 语义
        # 是子问题上限，切片要 +1 保留原始 query，否则永远少拆一个子问题。
        return result[:settings.rag.retrieval.multi_hop.max_subqueries + 1]

    except Exception as exc:
        # 子问题拆解失败不应阻断检索，但也不能静默吞掉；保留原始 query 并记录原因。
        print(f"[WARN] multi-hop 子问题拆解失败，回退仅用原始 query: {exc}")
        return [query]
