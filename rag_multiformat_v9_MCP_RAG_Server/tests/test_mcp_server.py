# -*- coding: utf-8 -*-
"""MCP 前端单元测试：不加载模型，仅覆盖工具注册、输出裁剪与 stdout 保护。"""
import asyncio
import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

try:
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.tools.base import ToolError
    HAS_MCP = True
except ImportError:
    HAS_MCP = False

from deployment import mcp_server as mod
from deployment.mcp_server import (
    _format_chat, _format_search, _guarded, _trim_contexts, _run_engine_call,
)


def _candidate(chunk_id, text, score=0.9, meta=None):
    return {
        "chunk_id": chunk_id,
        "collection": "rag_topology",
        "document": text,
        "rerank_score": score,
        "metadata": meta or {"relative_path": "电源设计/BUCK.pdf", "domain": "topology"},
    }


class TestFormatting(unittest.TestCase):
    def test_trim_contexts_caps_and_keeps_location(self):
        ctxs = [_candidate(str(i), f"chunk text {i}" * 100) for i in range(20)]
        trimmed = _trim_contexts(ctxs)
        self.assertEqual(len(trimmed), mod._CONTEXT_LIMIT)
        self.assertEqual(trimmed[0]["chunk_id"], "0")
        self.assertEqual(trimmed[0]["source_file"], "电源设计/BUCK.pdf")
        self.assertLessEqual(len(trimmed[0]["text"]), mod._TEXT_MAX_CHARS)

    def test_format_chat_keeps_answer_and_sources(self):
        payload = _format_chat({
            "answer": "答案是 42",
            "context_sufficient": True,
            "missing_information": [],
            "router_domains": ["topology"],
            "answer_sources": [
                {"label": "C1", "source_file": "a.pdf", "section": "2.1", "location": "page=3"},
            ],
            "selected_contexts": [_candidate("1", "正文")],
        })
        data = json.loads(payload)
        self.assertEqual(data["answer"], "答案是 42")
        self.assertEqual(data["sources"][0]["source_file"], "a.pdf")
        self.assertEqual(data["selected_contexts"][0]["chunk_id"], "1")

    def test_format_chat_generation_disabled_null_answer(self):
        # 服务端关闭生成：answer 为 null、generation_enabled 为 false，证据仍在 selected_contexts。
        payload = _format_chat({
            "answer": None,
            "generation_enabled": False,
            "context_sufficient": True,
            "selected_contexts": [_candidate("1", "正文")],
        })
        data = json.loads(payload)
        self.assertIsNone(data["answer"])
        self.assertFalse(data["generation_enabled"])
        self.assertEqual(data["selected_contexts"][0]["chunk_id"], "1")

    def test_format_chat_error_becomes_toolerror(self):
        with self.assertRaises(ToolError):
            _format_chat({"error": "retriever not loaded"})

    def test_format_search_caps_candidates(self):
        result = {"query": "q", "candidate_pool": [_candidate(str(i), f"t{i}") for i in range(50)]}
        data = json.loads(_format_search(result, "q"))
        self.assertEqual(data["count"], mod._SEARCH_TOP_K)
        self.assertIn("location", data["candidates"][0])

    def test_format_search_error_becomes_toolerror(self):
        with self.assertRaises(ToolError):
            _format_search({"error": "boom"}, "q")


class TestStdoutGuard(unittest.TestCase):
    def test_guarded_replays_leaked_prints_to_stderr(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            result = _guarded(lambda: print("泄漏输出") or "ok")
        self.assertEqual(result, "ok")
        self.assertEqual(buf.getvalue(), "")

    def test_run_engine_call_wraps_exceptions(self):
        def boom():
            raise RuntimeError("坏掉了")
        with self.assertRaises(ToolError) as ctx:
            _run_engine_call("测试", boom)
        self.assertIn("坏掉了", str(ctx.exception))

    def test_run_engine_call_wraps_systemexit(self):
        def die():
            raise SystemExit("空索引")
        with self.assertRaises(ToolError):
            _run_engine_call("测试", die)

    def test_run_engine_call_lock_timeout(self):
        # 锁被预热等长期占用时，工具调用应超时报错而不是无限排队。
        with patch.object(mod, "_CALL_LOCK_TIMEOUT", 0.05):
            mod._call_lock.acquire()
            try:
                with self.assertRaises(ToolError) as ctx:
                    _run_engine_call("测试", lambda: "x")
                self.assertIn("超时", str(ctx.exception))
            finally:
                mod._call_lock.release()


@unittest.skipUnless(HAS_MCP, "mcp SDK 未安装")
class TestToolRegistration(unittest.TestCase):
    def test_four_tools_registered_with_schemas(self):
        server = mod.server
        tools = asyncio.run(server.list_tools())
        names = {t.name for t in tools}
        self.assertTrue({"rag_chat", "rag_search", "rag_health", "rag_list_sources"} <= names)
        chat = next(t for t in tools if t.name == "rag_chat")
        self.assertEqual(chat.input_schema["required"], ["question"])

    def test_engine_stays_lazy_at_import(self):
        # 导入模块不应创建引擎（否则 stdio 握手前就会开始加载依赖）；
        # _get_engine() 只在工具首次被调用时执行。
        self.assertIsNone(mod._engine)


@unittest.skipUnless(HAS_MCP, "mcp SDK 未安装")
class TestAdminTools(unittest.TestCase):
    def setUp(self):
        self._orig_admin = mod._ADMIN_TOOLS
        mod._ADMIN_TOOLS = True
        mod.server = mod._build_server()

    def tearDown(self):
        mod._ADMIN_TOOLS = self._orig_admin
        mod._INGEST_PROC = None
        mod._INGEST_START_OFFSET = 0
        mod.server = mod._build_server()

    def test_ingest_single_flight(self):
        # 已有后台入库在跑时，第二次调用应复用进程，绝不并发启动第二个。
        class FakeProc:
            pid = 1234

            def poll(self):
                return None

            def wait(self):
                pass

        with patch.object(mod, "_run_subprocess_admin", return_value=FakeProc()) as fake:
            r1 = json.loads(asyncio.run(
                mod.server.call_tool("rag_ingest", {"wait": False})
            ).content[0].text)
            r2 = json.loads(asyncio.run(
                mod.server.call_tool("rag_ingest", {"wait": False})
            ).content[0].text)
        fake.assert_called_once()
        self.assertEqual(r1["pid"], 1234)
        self.assertEqual(r2["pid"], 1234)
        self.assertFalse(r1["done"])

    def test_ingest_new_run_after_finished(self):
        # 上一个进程已结束后再调用，应启动新进程并刷新日志起点。
        class FinishedProc:
            pid = 999

            def poll(self):
                return 0

            def wait(self):
                pass

        mod._INGEST_PROC = FinishedProc()
        mod._INGEST_START_OFFSET = 7
        with patch.object(mod, "_run_subprocess_admin", return_value=FinishedProc()) as fake:
            json.loads(asyncio.run(
                mod.server.call_tool("rag_ingest", {"wait": False})
            ).content[0].text)
        fake.assert_called_once()
        self.assertEqual(mod._INGEST_PROC.pid, 999)


if __name__ == "__main__":
    unittest.main()
