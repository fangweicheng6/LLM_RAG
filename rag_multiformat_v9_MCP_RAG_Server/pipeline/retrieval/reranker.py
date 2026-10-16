# -*- coding: utf-8 -*-
"""Optional CrossEncoder reranker."""
from __future__ import annotations
from typing import Any, Dict, List
from utils.config_loader import settings

_model = None

def load_reranker_model():
    """加载（并缓存）reranker 模型，供启动预加载与在线 rerank 复用。"""
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        # 优先离线加载本地缓存，避免访问 HuggingFace 触发未认证告警。
        try:
            _model = CrossEncoder(
                settings.model.reranker.name,
                local_files_only=True,
            )
        except Exception:
            _model = CrossEncoder(settings.model.reranker.name)
    return _model

def rerank_candidates(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not settings.rag.retrieval.rerank.enabled or not candidates:
        return candidates
    try:
        model = load_reranker_model()
        pairs=[(query, str(x.get("document",""))) for x in candidates]
        scores=model.predict(pairs)
        for x,score in zip(candidates,scores):
            score=float(score)
            x["rerank_score"]=score
            # 重排后的相关度必须同步回 similarity/distance，保持 similarity=1-distance 约定。
            # 否则下游 selector 读到的仍是重排前过期的余弦相似度，排序与信号脱节。
            x["similarity"]=score
            x["distance"]=1.0-score
            # 保留原始召回来源（vector/lexical），避免被 "reranker" 覆盖后丢失溯源。
            x.setdefault("pre_rerank_source", x.get("retrieval_source","vector"))
            x["retrieval_source"]="reranker"
        return sorted(candidates,key=lambda x:x.get("rerank_score",-1.0),reverse=True)[:settings.rag.retrieval.rerank.top_k]
    except Exception as e:
        # reranker故障不影响基础RAG
        for x in candidates:
            x["rerank_error"]=str(e)
        return candidates
