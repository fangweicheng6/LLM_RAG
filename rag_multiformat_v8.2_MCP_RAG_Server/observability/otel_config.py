try:
    from opentelemetry import trace
    tracer=trace.get_tracer("industrial-rag")
except Exception:
    tracer=None

def start_span(name):
    if tracer:
        return tracer.start_as_current_span(name)
    return None
