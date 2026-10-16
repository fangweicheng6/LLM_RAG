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


def generate_final_answer(
    *,
    query,
    selected_contexts,
    llm,
    context_sufficient,
    missing_information,
    generator,
):
    """最终回答阶段的开关（rag.generation.enabled）。

    关闭时完全不进入回答生成：返回 ``(None, [])``，响应里的 ``answer`` 为显式
    ``null``，相关 chunk 仍由 ``selected_contexts`` 原样提供，供外部系统用自己的
    LLM 对接。开启时行为与本开关不存在时完全一致。

    注意：影响所有经过 ``retriever._run`` 的入口 —— 在线服务（HTTP / Socket /
    MCP 的 /chat 家族）与 E2E 评测（evaluation/rag_valide，即 benchmark 的
    e2e 分支）。生成关闭时 E2E 的 answer 相关指标按空答案记 0；需要评估生成
    质量时把开关置 true。generation 专项基准
    （evaluation/benchmark_evaluators.evaluate_generation）直接调用
    ``generate_answer``，不受该开关影响。
    """
    if not settings.rag.generation.enabled:
        return None, []
    return generator(
        query=query,
        selected_contexts=selected_contexts,
        llm=llm,
        context_sufficient=context_sufficient,
        missing_information=missing_information,
    )


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
