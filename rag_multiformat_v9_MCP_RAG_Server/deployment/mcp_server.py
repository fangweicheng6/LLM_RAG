# -*- coding: utf-8 -*-
"""MCP (Model Context Protocol) frontend for the Industrial RAG service.

All frontends (HTTP / Socket / MCP) call the same RAGEngine
(``server.runtime.service_runtime.create_service``)。

两种传输方式:

1. stdio —— 供 Claude Desktop / Claude Code / Cursor 等本地 MCP 客户端使用:

       python -m deployment.mcp_server
       # 或: python -m server.start_server --mode mcp

2. Streamable HTTP —— 供远程 MCP 客户端使用:

       python -m deployment.mcp_server --transport streamable-http --port 8100
       # 或: python -m server.start_server --mode mcp-http

设计要点:

- **懒加载引擎**: MCP 客户端要求服务端快速完成 initialize 握手，
  启动时不预加载 embedding/reranker 模型
  （``create_service(preload=False)``），改为后台线程预热；
  首个查询前模型通常已就绪。
- **stdout 保护**: stdio 传输中 stdout 是 MCP 协议通道，任何 print
  都会破坏协议帧。所有引擎调用都包在 ``redirect_stdout`` 里，
  泄漏的打印改走 stderr。
- **调用串行化**: 引擎内部的 embedding 模型 / Chroma 客户端 / LLM
  客户端是模块级共享状态，不保证并发安全；所有引擎调用共用一把
  锁串行执行（MCP 场景通常单人使用，串行化代价可接受）。
- **异常兜底**: 空索引等场景下管线内部会 raise SystemExit
  （utils.index_audit），若任其在工作线程里逃逸会杀死整个 MCP
  进程；这里捕获 BaseException 并转为客户端可见的 ToolError。
- **任意 cwd 可启动**: 文件顶部做了 sys.path 兜底，客户端以
  python.exe 为 command、本文件为 args 启动（不依赖工作目录）也能
  正常导入项目包。
- **管理工具按需开启**: rag_ingest / rag_index_audit 默认关闭，
  ``service.mcp.expose_admin_tools: true`` 时暴露（入库会改写索引与
  向量库，需明确授权）。
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# 直接以脚本方式运行（如 Claude Desktop 配置里 command 为 python.exe、
# args 为本文件）时，项目根目录不在 sys.path 上；这里兜底插入，
# 保证任意 cwd 都能导入项目包。
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# stderr 日志统一 UTF-8：Windows 控制台默认 cp936，日志里的生僻字
# 可能导致 UnicodeEncodeError 中断服务端；协议通道(stdout)由 SDK
# 按字节写，不受影响。
try:
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

try:
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.tools.base import ToolError
except ImportError:  # mcp 未安装时给出可读错误，而不是 traceback
    MCPServer = None
    ToolError = RuntimeError

from utils.config_loader import settings


# ============================================================
# 懒加载引擎 + 后台预热
# ============================================================

_engine = None
_engine_lock = threading.Lock()
_call_lock = threading.Lock()  # 引擎调用串行化（见模块 docstring）


def _get_engine():
    global _engine
    with _engine_lock:
        if _engine is None:
            from server.runtime.service_runtime import create_service
            _engine = create_service(preload=False)
        return _engine


def _start_warmup():
    """后台线程预热 embedding/reranker/向量库，不阻塞 MCP 握手。

    MCP 客户端对 initialize 响应有超时预期，模型加载动辄几十秒，
    放主流程会表现为“MCP 连接失败”。预热失败只影响首个查询的
    冷启动延迟，不影响服务可用性。
    """
    if os.environ.get("RAG_MCP_WARMUP", "").strip().lower() in ("0", "false", "no"):
        return
    if not settings.get("service.mcp.warmup", True):
        return

    def _warmup():
        try:
            with _call_lock:  # 与首次查询互斥，避免模型被并发加载两次
                _get_engine()
                # 不在这里 redirect_stdout：stdio 传输建立后 SDK 会把 fd 1
                # 分流到 stderr（mcp/server/stdio.py 的 _claim_fd），preload 的
                # 进度打印自动落在 stderr、不会污染协议通道；在 claim 之前
                # 改写 sys.stdout 反而会与传输层的 fd 抢占产生竞态。
                from server.runtime.preload import preload_runtime
                preload_runtime()
        except Exception as exc:
            print(f"[MCP] 后台预热失败（首个查询会自行加载）: {exc}",
                  file=sys.stderr)

    threading.Thread(target=_warmup, name="mcp-warmup", daemon=True).start()


def _guarded(call):
    """在 stdout 重定向下执行引擎调用，防止协议通道被 print 污染。"""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = call()
    leaked = buffer.getvalue().strip()
    if leaked:
        print("[MCP] 引擎输出已重定向到 stderr:", file=sys.stderr)
        print(leaked, file=sys.stderr)
    return result


def _trace(call):
    """带 @trace_rag 的引擎调用；等价于 HTTP 前端的可观测性边界。"""
    from observability import trace_rag
    return trace_rag(lambda: call())()


_CALL_LOCK_TIMEOUT = float(settings.get("service.mcp.call_lock_timeout", 120))


def _run_engine_call(action: str, call):
    """工具执行统一入口：限时串行化 -> 可观测性 -> stdout 保护 -> 异常转 ToolError。

    串行化限时获取（而非无限等待）：预热线程会在模型加载期间长期持有
    _call_lock，若加载卡死（如在线回退挂起），工具调用在超时后以明确的
    ToolError 失败，而不是永久排队。
    BaseException 也要捕获：空索引时管线内部（utils.index_audit）会
    raise SystemExit，任其逃逸会杀死 MCP 进程。
    """
    if not _call_lock.acquire(timeout=_CALL_LOCK_TIMEOUT):
        raise ToolError(
            f"{action} 超时：引擎调用排队超过 {_CALL_LOCK_TIMEOUT:.0f} 秒"
            "（模型预热或前一调用未完成），请稍后重试"
        )
    try:
        return _guarded(lambda: _trace(call))
    except ToolError:
        raise
    except BaseException as exc:
        raise ToolError(f"{action} 失败: {exc}") from exc
    finally:
        _call_lock.release()


# ============================================================
# 工具输出裁剪（候选池可达上百块，避免撑爆客户端上下文）
# ============================================================

_SEARCH_TOP_K = 20
_CONTEXT_LIMIT = 8
_TEXT_MAX_CHARS = 2000


def _trim_contexts(contexts):
    out = []
    for c in (contexts or [])[:_CONTEXT_LIMIT]:
        meta = c.get("metadata") or {}
        out.append({
            "chunk_id": str(c.get("chunk_id", "")),
            "collection": c.get("collection", ""),
            "source_file": meta.get("relative_path", "") or c.get("source_file", ""),
            "domain": meta.get("domain", ""),
            "score": c.get("rerank_score", c.get("similarity")),
            "text": (c.get("document") or "")[:_TEXT_MAX_CHARS],
        })
    return out


def _format_chat(result):
    if not isinstance(result, dict):
        return json.dumps({"answer": str(result)}, ensure_ascii=False, indent=2)
    if result.get("error"):
        raise ToolError(str(result["error"]))
    sources = []
    for s in result.get("answer_sources") or []:
        sources.append({
            "label": s.get("label", ""),
            "source_file": s.get("source_file", ""),
            "section": s.get("section", ""),
            "location": s.get("location", ""),
        })
    payload = {
        "answer": result.get("answer", ""),
        # 服务端关闭生成（configs/rag.yaml: rag.generation.enabled=false）时为 false，
        # 此时 answer 为 null，证据仍完整地放在 selected_contexts 里。
        "generation_enabled": bool(result.get("generation_enabled", True)),
        "context_sufficient": result.get("context_sufficient"),
        "missing_information": result.get("missing_information", []),
        "router_domains": result.get("router_domains", []),
        "domains": result.get("domains", []),
        "selection_reason": result.get("selection_reason", ""),
        "sources": sources,
        "selected_contexts": _trim_contexts(result.get("selected_contexts")),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _format_search(result, query):
    if not isinstance(result, dict):
        return json.dumps({"candidates": []}, ensure_ascii=False, indent=2)
    if result.get("error"):
        raise ToolError(str(result["error"]))

    try:
        from ingestion.source_store import format_location
    except ImportError:
        format_location = lambda meta: ""

    items = []
    for c in (result.get("candidate_pool") or [])[:_SEARCH_TOP_K]:
        meta = c.get("metadata") or {}
        items.append({
            "chunk_id": str(c.get("chunk_id", "")),
            "collection": c.get("collection", ""),
            "score": c.get("rerank_score", c.get("similarity")),
            "source_file": meta.get("relative_path", ""),
            "domain": meta.get("domain", ""),
            "location": format_location(meta),
            "text": (c.get("document") or "")[:_TEXT_MAX_CHARS],
        })
    payload = {
        "query": result.get("query", query),
        "router_domains": result.get("router_domains", []),
        "domains": result.get("domains", []),
        "count": len(items),
        "candidates": items,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


# ============================================================
# MCP Server 与工具注册
# ============================================================

_ADMIN_TOOLS = (
    settings.get("service.mcp.expose_admin_tools", False)
    or os.environ.get("RAG_MCP_ADMIN_TOOLS", "").strip().lower() in ("1", "true", "yes")
)


def _ingest_log_path() -> Path:
    return _PROJECT_ROOT / settings.get("paths.data_dir", "data_index_only") / "ingest_mcp.log"


def _run_subprocess_admin(args, *, log_path: Path | None):
    """以子进程方式执行离线命令（入库/索引审计），避免长任务占住工具调用。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_stream = log_path.open("a", encoding="utf-8") if log_path else subprocess.DEVNULL
    proc = subprocess.Popen(
        [sys.executable, "-m", *args],
        cwd=_PROJECT_ROOT,
        stdout=log_stream,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    if log_stream is not subprocess.DEVNULL:
        log_stream.close()
    return proc


# 入库单飞控制：同一 MCP 进程内只允许一个入库子进程。wait=True 时等待的是
# 先前启动的后台任务而不是再起一个，避免两个进程同时写 Chroma 库与切片文件。
_INGEST_PROC = None
_INGEST_START_OFFSET = 0
_INGEST_LOCK = threading.Lock()


def _build_server() -> MCPServer:
    server = MCPServer(
        name="industrial-rag",
        version="8.2.0",
        instructions=(
            "工业级多格式 RAG 知识库检索服务。\n"
            "工具说明：\n"
            "- rag_chat: 完整 RAG 问答（检索 + 上下文选择 + LLM 生成），"
            "回答可追溯到具体出处，适合直接回答用户问题。\n"
            "- rag_search: 仅检索不生成，返回带全文与位置的候选块，"
            "适合需要自行判断证据、或 rag_chat 回答不充分时深挖原文。\n"
            "- rag_health: 服务健康检查。\n"
            "- rag_list_sources: 列出各领域向量集合与块数量。\n"
            + (
                "- rag_ingest: 离线更新知识库。默认全量（重新解析全部文件）；"
                "incremental=true 只重解析新增/内容变化的文件并删除已删文件的 chunk。"
                "两种模式都会把向量库对账到索引，执行期间查询结果可能不完整。\n"
                "- rag_index_audit: 索引一致性审计。\n"
                if _ADMIN_TOOLS else ""
            ) +
            "注意：\n"
            "- 答案完全基于知识库；知识库不充分时 rag_chat 会明确"
            "说明缺失信息而不是编造。\n"
            "- 服务依赖环境变量 LLM_API_KEY 调用生成模型。\n"
            "- 若服务端关闭了生成（configs/rag.yaml: rag.generation.enabled=false），"
            "rag_chat 返回的 answer 为 null、generation_enabled 为 false，"
            "证据仍完整保留在 selected_contexts 中；此时可直接使用 rag_search，"
            "或由客户端自己的 LLM 基于这些 chunk 生成回答。"
        ),
    )

    @server.tool(structured_output=False)
    def rag_chat(question: str) -> str:
        """完整 RAG 问答：检索知识库 -> LLM 上下文选择 -> 生成可追溯答案。

        基于知识库内容回答用户问题，返回答案、出处和置信信息。
        当知识库不足以可靠回答时，会明确说明缺失信息而不编造。

        Args:
            question: 用户问题，中文或英文均可。

        Returns:
            JSON 字符串，字段：answer（答案；服务端关闭生成时为 null）、
            generation_enabled（本响应是否经过回答生成阶段）、
            context_sufficient（上下文是否充分）、
            missing_information（缺失信息）、router_domains（路由领域）、
            sources（答案出处）、selected_contexts（被选中的上下文块，含全文）。
        """
        question = (question or "").strip()
        if not question:
            raise ToolError("question 不能为空")

        def call():
            engine = _get_engine()
            return _format_chat(engine.chat(question))

        return _run_engine_call("rag_chat", call)

    @server.tool(structured_output=False)
    def rag_search(query: str) -> str:
        """仅检索不生成：返回与查询相关的知识库候选块（含全文与出处位置）。

        与 rag_chat 不同，本工具不调用生成模型，返回原始候选块，
        适合验证证据、补充检索或分析知识库内容。

        Args:
            query: 检索查询，中文或英文均可。

        Returns:
            JSON 字符串，字段：query、router_domains、count、
            candidates（每个含 chunk_id、score、source_file、
            domain、location、text）。
        """
        query = (query or "").strip()
        if not query:
            raise ToolError("query 不能为空")

        def call():
            engine = _get_engine()
            return _format_search(engine.search(query), query)

        return _run_engine_call("rag_search", call)

    @server.tool(structured_output=False)
    def rag_health() -> str:
        """服务健康检查：返回已加载的组件列表与状态。"""
        def call():
            engine = _get_engine()
            return json.dumps(engine.health(), ensure_ascii=False, indent=2)

        return _run_engine_call("rag_health", call)

    @server.tool(structured_output=False)
    def rag_list_sources() -> str:
        """列出知识库各领域的向量集合与块数量，了解知识库覆盖范围。"""
        def call():
            from pipeline.retrieval.retriever import get_client
            client = get_client()
            prefix = settings.paths.collection_prefix
            rows = []
            for coll in sorted(client.list_collections(), key=lambda c: c.name):
                if not coll.name.startswith(prefix):
                    continue
                try:
                    count = coll.count()
                except Exception:
                    count = None
                rows.append({"collection": coll.name, "chunks": count})
            return json.dumps(
                {"collection_prefix": prefix, "collections": rows},
                ensure_ascii=False, indent=2,
            )

        return _run_engine_call("rag_list_sources", call)

    if _ADMIN_TOOLS:

        @server.tool(structured_output=False)
        def rag_index_audit() -> str:
            """索引一致性审计：校验向量库结构、metadata 白名单与原文存储隔离。

            返回审计命令的输出摘要与退出码（0 表示通过）。
            """
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "utils.index_audit"],
                    cwd=_PROJECT_ROOT,
                    capture_output=True,
                    timeout=300,
                    env={**os.environ, "PYTHONUTF8": "1"},
                )
            except subprocess.TimeoutExpired:
                raise ToolError("索引审计超时（300 秒）")
            # SystemExit 等失败信息走 stderr，只保留 stdout 会让最常见的
            # 失败场景（空索引）返回空的 output_tail，客户端看不到原因。
            stdout_tail = (proc.stdout or b"").decode("utf-8", errors="replace")
            stderr_tail = (proc.stderr or b"").decode("utf-8", errors="replace")
            output = ("STDERR:\n" + stderr_tail + "\nSTDOUT:\n" + stdout_tail).strip()[-4000:]
            return json.dumps(
                {"ok": proc.returncode == 0, "returncode": proc.returncode,
                 "output_tail": output},
                ensure_ascii=False, indent=2,
            )

        @server.tool(structured_output=False)
        def rag_ingest(
            incremental: bool = False,
            wait: bool = False,
            timeout_seconds: int = 1800,
        ) -> str:
            """离线更新知识库：扫描知识库 -> 多格式解析 -> 切片 ->
            Embedding -> 把向量库对账到索引
            （等价于 python -m ingestion.ingest [--incremental]）。

            两种模式的差别只在切片阶段：
            - incremental=true：只重新解析新增/内容变化的文件，并删除已从知识库
              移除的文件的 chunk；只处理变化的文件，通常几十秒到几分钟。
            - incremental=false（默认）：重新解析全部文件，耗时可达数十分钟。
            两者都会把向量库对账到索引（多退少补），执行期间查询结果可能不完整。
            wait=false 时立即返回，任务在后台子进程运行，日志写入
            data_index_only/ingest_mcp.log；wait=true 时同步等待完成
            并返回本次运行的日志尾部。同一服务进程内同时只运行一个入库
            任务：若已有任务在跑，wait=true 会等待该任务而不是再起一个。

            Args:
                incremental: 是否只重解析新增/变化的文件（默认 false）。
                wait: 是否同步等待入库完成（默认 false）。
                timeout_seconds: wait=true 时的最长等待秒数（默认 1800）。

            Returns:
                JSON 字符串，字段：pid、log_file、done、returncode、
                log_tail。
            """
            global _INGEST_PROC, _INGEST_START_OFFSET
            timeout_seconds = max(60, min(int(timeout_seconds or 1800), 7200))
            log_path = _ingest_log_path()
            with _INGEST_LOCK:
                proc = _INGEST_PROC
                if proc is not None and proc.poll() is None:
                    # 已有后台入库在跑：复用它的进程与日志起点，绝不并发启动第二个
                    start_offset = _INGEST_START_OFFSET
                else:
                    if proc is not None:
                        proc.wait()  # 回收已结束的进程，避免僵尸
                    _INGEST_PROC = None
                    args = ["ingestion.ingest"]
                    if incremental:
                        args.append("--incremental")
                    try:
                        proc = _run_subprocess_admin(args, log_path=log_path)
                    except Exception as exc:
                        raise ToolError(f"入库子进程启动失败: {exc}")
                    start_offset = log_path.stat().st_size if log_path.exists() else 0
                    _INGEST_START_OFFSET = start_offset
                    _INGEST_PROC = proc
            if not wait:
                return json.dumps(
                    {"pid": proc.pid, "log_file": str(log_path), "done": False,
                     "returncode": None, "log_tail": ""},
                    ensure_ascii=False, indent=2,
                )
            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    break
                time.sleep(2)
            done = proc.poll() is not None
            if done:
                with _INGEST_LOCK:
                    if _INGEST_PROC is proc:
                        _INGEST_PROC = None
            tail = ""
            if log_path.exists():
                try:
                    # 按字节偏移只取本次运行新增的日志，避免混入上一次入库的旧输出
                    tail = log_path.read_bytes()[start_offset:][-4000:].decode(
                        "utf-8", errors="replace"
                    )
                except OSError:
                    pass
            return json.dumps(
                {"pid": proc.pid, "log_file": str(log_path), "done": done,
                 "returncode": proc.returncode if done else None, "log_tail": tail},
                ensure_ascii=False, indent=2,
            )

    return server


server = _build_server() if MCPServer is not None else None


# ============================================================
# 入口
# ============================================================

def main(argv=None):
    # parser 先于 SDK 可用性检查构建：mcp 未安装时 --help 仍应可用。
    parser = argparse.ArgumentParser(description="Industrial RAG MCP server")
    parser.add_argument(
        "--transport", choices=["stdio", "streamable-http"], default="stdio",
        help="传输方式: stdio（本地客户端）或 streamable-http（远程客户端）",
    )
    parser.add_argument("--host", default=None,
                        help="streamable-http 监听地址（默认取 service.mcp.http_host）")
    parser.add_argument("--port", type=int, default=None,
                        help="streamable-http 监听端口（默认取 service.mcp.http_port）")
    args = parser.parse_args(argv)

    if server is None:
        print(
            "未安装 MCP SDK，无法启动 MCP 服务。\n"
            "请先执行: pip install mcp==2.3.0\n"
            "或完整安装: pip install -r requirements.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    _start_warmup()

    if args.transport == "stdio":
        server.run(transport="stdio")
        return

    host = args.host or settings.get("service.mcp.http_host", "127.0.0.1")
    port = args.port or int(settings.get("service.mcp.http_port", 8100))
    path = settings.get("service.mcp.streamable_http_path", "/mcp")
    print(f"MCP streamable-http: http://{host}:{port}{path}", file=sys.stderr)
    server.run(transport="streamable-http", host=host, port=port,
               streamable_http_path=path)


if __name__ == "__main__":
    main()
