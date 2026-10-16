
# -*- coding: utf-8 -*-
"""
Unified RAG execution engine.

All frontends (HTTP / Socket / future MCP) call this engine.
Architecture:

server
  -> RAGEngine
       -> Router
       -> Retriever
       -> Generator
       -> Observability
"""
from typing import Any, Dict


class RAGEngine:
    def __init__(self, registry):
        self.registry = registry

    def health(self):
        return {
            "status": "ok",
            "components": self.registry.status(),
        }

    def chat(self, question: str) -> Dict[str, Any]:
        router = self.registry.get("router")
        retriever = self.registry.get("retriever")
        generator = self.registry.get("generator")

        if retriever is None:
            return {"error": "retriever not loaded"}

        # Current v7.16 search pipeline is exposed as a component.
        # The engine owns the call boundary; frontends no longer know search.py.
        result = retriever(question)

        # 检索管线已产出 answer 字段即视为回答阶段结束：正常生成、确定性拒答、
        # 以及「生成已关闭」的显式 null 都直接返回。
        # 不再用真值判断（旧逻辑会把空串/null 当作“未生成”再走 Generator 兜底），
        # 避免把“故意不生成”误判为“生成失败”而触发第二次生成。
        if isinstance(result, dict) and "answer" in result:
            return result

        if generator is None:
            return result if isinstance(result, dict) else {"result": result}

        answer = generator(question, result)
        if isinstance(result, dict):
            result["answer"] = answer
            return result

        return {"answer": answer, "result": result}

    def search(self, query: str):
        retriever = self.registry.get("retriever")
        if retriever is None:
            return {"error": "retriever not loaded"}
        return retriever(query, retrieval_only=True)
