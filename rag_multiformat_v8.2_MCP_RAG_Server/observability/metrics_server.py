from prometheus_client import Counter, Histogram, generate_latest

RAG_REQUESTS = Counter("rag_requests_total", "Total RAG requests")
RAG_LATENCY = Histogram("rag_request_latency_seconds", "RAG latency")
RAG_ERRORS = Counter("rag_errors_total", "RAG errors")

def metrics_text():
    return generate_latest()
