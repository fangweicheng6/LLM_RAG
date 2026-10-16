from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

from server.runtime.service_runtime import create_service
from observability import trace_rag
from observability.metrics_server import metrics_text

app=FastAPI(title="Industrial RAG Service v7.16")

engine=create_service()

class Query(BaseModel):
    question:str

class SearchQuery(BaseModel):
    query:str

@app.get("/health")
def health():
    return engine.health()

@app.get("/metrics")
def metrics():
    return Response(metrics_text(), media_type="text/plain")

@app.post("/chat")
@trace_rag
def chat(req:Query):
    # 延迟由 @trace_rag 统一记录（RAG_LATENCY.observe），这里不要再套 RAG_LATENCY.time()，
    # 否则同一请求的延迟会被记两次，拉高直方图计数。
    return engine.chat(req.question)

@app.post("/search")
@trace_rag
def search(req:SearchQuery):
    # 纯检索接口：不生成回答，返回候选块（等价于 MCP 的 rag_search）。
    # 候选块在 candidate_pool 中，已含全文与出处定位，供外部系统对接自己的 LLM。
    return engine.search(req.query)
