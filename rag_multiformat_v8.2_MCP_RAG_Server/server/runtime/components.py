
# -*- coding: utf-8 -*-
"""Runtime adapters for unified RAGEngine."""

from pipeline.retrieval.retriever import run, run_retrieval


class RetrieverAdapter:
    def __call__(self, query, retrieval_only=False):
        if retrieval_only:
            return run_retrieval(query, quiet=True)
        # 检索未命中不应让 HTTP 层返回 500：run 在 allow_empty=False 时会把
        # 空召回当作 RuntimeError 抛出。这里允许空结果，返回优雅的
        # “无法可靠回答”而非 500。
        return run(query, quiet=True, allow_empty=True)


class GeneratorAdapter:
    """
    Kept as an extension point.
    The v7.16 search pipeline already invokes generator internally.
    """
    def __call__(self, query, retrieval_result):
        return retrieval_result.get("answer", "") if isinstance(retrieval_result, dict) else ""
