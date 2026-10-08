"""路由与上下文选择的独立策略；不依赖 Ragas，不改变原来的命令行入口。"""
from __future__ import annotations

from utils.config_loader import settings


def route_query(query, llm, available_domains, *, analyzer):
    if settings.rag.router.enabled:
        return analyzer(query=query, llm=llm)
    return {
        "keywords": [],
        "domains": list(available_domains),
        "reason": "Router 已关闭：检索全部现有知识域；不调用关键词提取或路由 LLM。",
    }


def choose_contexts(*, query, candidates, llm, allow_expansion, selector):
    if settings.rag.context.selector.enabled:
        return selector(query=query, candidates=candidates, llm=llm,
                        allow_expansion=allow_expansion)
    k = settings.rag.retrieval.top_k
    if k <= 0:
        raise ValueError("settings.rag.retrieval.top_k 必须大于 0")
    def distance(item):
        value = item.get("distance")
        return float(value) if isinstance(value, (int, float)) else float("inf")
    selected = sorted(candidates, key=distance)[:k]
    return {
        "selected_chunk_ids": [item["chunk_id"] for item in selected],
        # 关闭选择器时不伪造“LLM 已确认充分”。沿用保守的答案生成提示。
        "sufficient": False,
        "missing_information": [],
        "expansion_requests": [],
        "reason": f"LLM Chunk Selector 已关闭：使用向量距离 Top-{k}，未判断充分性。",
    }, selected
