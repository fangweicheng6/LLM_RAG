# -*- coding: utf-8 -*-
"""
工业RAG可观测性统一入口
- Langfuse: LLM/RAG Trace
- OpenTelemetry: 分布式链路
- Prometheus: 指标
"""

import os, time, functools, contextlib, contextvars
from utils.config_loader import settings

LANGFUSE_ENABLED=settings.get("service.observability.langfuse.enabled", False)

try:
    from observability.metrics_server import RAG_REQUESTS, RAG_LATENCY
except Exception:
    RAG_REQUESTS=None
    RAG_LATENCY=None

try:
    from langfuse import Langfuse
    langfuse=Langfuse() if LANGFUSE_ENABLED else None
except Exception:
    langfuse=None

_current_trace = contextvars.ContextVar("langfuse_trace", default=None)

def current_trace():
    return _current_trace.get()

@contextlib.contextmanager
def langfuse_span(name, input=None, output=None):
    """
    Langfuse v4 child span（上下文管理器）。

    用 ``with langfuse_span(...) as span:`` 包裹业务代码，
    span 会在 ``with`` 退出时（含异常路径）在同一线程自动关闭，
    避免 context token 泄漏到其他线程被 GC。
    """
    if not langfuse:
        yield None
        return
    try:
        cm = langfuse.start_as_current_observation(
            name=name,
            as_type="span",
            input=input,
            output=output,
        )
    except Exception as e:
        print(f"Langfuse span create error({name}):", repr(e))
        yield None
        return
    with cm as span:
        yield span

def update_span(span, output=None):
    if span is None:
        return
    try:
        span.update(output=output)
    except Exception:
        pass

def trace_rag(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        if RAG_REQUESTS:
            RAG_REQUESTS.inc()

        try:
            if not langfuse:
                return func(*args, **kwargs)

            try:
                cm = langfuse.start_as_current_observation(
                    name="rag_query",
                    as_type="span",
                    input={"args": str(args)[:1000], "kwargs": str(kwargs)[:1000]},
                )
            except Exception as e:
                print("Langfuse observation error:", repr(e))
                return func(*args, **kwargs)

            # 用 `with` 保证 __enter__ / __exit__ 在同一线程成对执行，
            # 否则 OpenTelemetry 的 context token 泄漏到其他线程被 GC 时
            # 会报 "was created in a different Context"。
            with cm as trace:
                token = _current_trace.set(trace)
                try:
                    result = func(*args, **kwargs)
                    try:
                        trace.update(
                            output=result
                            if isinstance(result, dict)
                            else str(result)
                        )
                    except Exception:
                        pass
                    return result
                finally:
                    if token is not None:
                        _current_trace.reset(token)
        finally:
            if RAG_LATENCY:
                RAG_LATENCY.observe(time.time() - start)
            if langfuse:
                try:
                    langfuse.flush()
                except Exception as e:
                    print("Langfuse flush error:", repr(e))

    return wrapper

def health():
    return {"observability":"enabled","langfuse":bool(langfuse)}
