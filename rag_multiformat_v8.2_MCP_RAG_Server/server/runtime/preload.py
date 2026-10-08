# -*- coding: utf-8 -*-
"""启动期预加载 embedding / reranker / Chroma，避免首个请求冷启动。"""

from utils.config_loader import settings


def preload_runtime():
    if settings.get("service.runtime.preload_models", True):
        from pipeline.retrieval.retriever import load_embedding_model
        load_embedding_model()

        if settings.get("rag.retrieval.rerank.enabled", False):
            from pipeline.retrieval.reranker import load_reranker_model
            load_reranker_model()

    if settings.get("service.runtime.preload_vector_db", True):
        from pipeline.retrieval.retriever import get_client
        get_client()
