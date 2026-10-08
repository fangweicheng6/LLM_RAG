
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

        # If retriever already completed generation (compatibility mode)
        if isinstance(result, dict) and result.get("answer"):
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
