# -*- coding: utf-8 -*-
"""交互式 MCP 客户端：连上 RAG MCP 服务，菜单选工具 -> 执行 -> 循环，可反复调用。

用法（服务端需先启动，如 start_mcp.bat 监听 http://<host>:8100/mcp）:

    python -m client.mcp_client                              # 默认连 192.168.124.127:8100
    python -m client.mcp_client --url http://127.0.0.1:8100/mcp

交互说明:
- 启动时向服务端拉取已注册工具清单，打印编号 + 简介（连不上则回退到内置
  工具列表，调用时仍会重试连接）。
- 需要参数的工具（rag_chat 的 question、rag_search 的 query）会提示输入；
  不需要参数的工具（rag_health / rag_list_sources）选中即执行。
- 每次执行完回到菜单可继续调用；输入 q 退出。
- rag_chat 依赖服务端环境变量 LLM_API_KEY；知识库需先入库
  （python -m ingestion.ingest）。
"""
import argparse
import asyncio
import json
import sys

DEFAULT_URL = "http://192.168.124.127:8100/mcp"

# 菜单简介：服务端 description 较长，这里给一句人话总结。
# 服务端新增的工具不在表里时，自动改用 description 首行。
TOOL_INTROS = {
    "rag_chat": "完整 RAG 问答：检索 + 上下文选择 + LLM 生成（服务端关闭生成最后答案），答案可追溯到出处",
    "rag_search": "仅检索不生成：返回相关候选块（含全文与出处位置），适合查证据",
    "rag_health": "服务健康检查：返回已加载的组件与状态",
    "rag_list_sources": "列出各领域向量集合与块数量，了解知识库覆盖范围",
    "rag_ingest": "离线更新知识库（全量或 incremental 增量，都会把向量库对账到索引）",
    "rag_index_audit": "索引一致性审计（校验向量库结构与 metadata 白名单）",
}

# 参数名 -> 输入提示语（问题类参数统一提示“请输入问题”）
PARAM_PROMPTS = {
    "question": "请输入问题",
    "query": "请输入问题",
}

# 拿不到服务端 schema 时的兜底必填参数（保证不会漏提示导致调用被服务端拒绝）
KNOWN_REQUIRED = {
    "rag_chat": ["question"],
    "rag_search": ["query"],
}


# ------------------------------------------------------------
# MCP 会话（每次调用单独建连接，对应你原来脚本的写法）
# ------------------------------------------------------------

async def _list_tools(url):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with streamable_http_client(url) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            return list((await session.list_tools()).tools)


async def _call_tool(url, tool_name, arguments):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with streamable_http_client(url) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            text = "\n".join(getattr(c, "text", str(c)) for c in (result.content or []))
            if getattr(result, "is_error", False):
                return f"[服务器返回错误]\n{text}"
            return text


# ------------------------------------------------------------
# 工具清单 / 参数提示
# ------------------------------------------------------------

class _FallbackTool:
    """拉取工具清单失败时的占位，字段与 mcp 的 Tool 对象对齐。"""

    def __init__(self, name, input_schema):
        self.name = name
        self.description = TOOL_INTROS.get(name, "")
        self.input_schema = input_schema


_FALLBACK_TOOLS = [
    _FallbackTool("rag_chat", {"properties": {"question": {}}, "required": ["question"]}),
    _FallbackTool("rag_health", {}),
    _FallbackTool("rag_list_sources", {}),
    _FallbackTool("rag_search", {"properties": {"query": {}}, "required": ["query"]}),
]


def _schema(tool):
    # mcp 2.3.0 客户端字段为 snake_case；兼容 camelCase 写法
    return getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {}


def _required_params(tool):
    schema = _schema(tool)
    props = schema.get("properties") or {}
    required = schema.get("required")
    if required is None:
        required = KNOWN_REQUIRED.get(tool.name, [])
    return [(name, props.get(name) or {}) for name in required]


def _intro(tool):
    text = TOOL_INTROS.get(tool.name)
    if not text:
        text = (getattr(tool, "description", "") or "").strip()
        text = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return text


def _print_menu(tools):
    print("\n可用工具（选择后会立即执行，需要参数的工具会再提示输入）：")
    for i, tool in enumerate(tools, 1):
        params = [name for name, _ in _required_params(tool)]
        hint = "需要参数：" + "、".join(params) if params else "无需参数"
        print(f"  {i}. {tool.name}（{hint}）")
        print(f"     {_intro(tool)}")
    print("  q. 退出")


def _ask(prompt):
    try:
        return input(prompt).strip()
    except EOFError:  # 管道输入结束时当作退出
        return "q"


def _coerce(value, schema):
    """按 schema 类型把输入文本转成对应类型，转换失败保持字符串原样。"""
    kind = schema.get("type")
    try:
        if kind == "integer":
            return int(value)
        if kind == "number":
            return float(value)
        if kind == "boolean":
            return value.lower() in ("1", "true", "yes", "y", "是")
    except ValueError:
        pass
    return value


# ------------------------------------------------------------
# 交互循环
# ------------------------------------------------------------

def _interactive(url, tools):
    while True:
        _print_menu(tools)
        choice = _ask("\n请选择工具（编号，q 退出）: ")

        if choice.lower() in ("q", "quit", "exit"):
            break

        tool = None
        if choice.isdigit() and 1 <= int(choice) <= len(tools):
            tool = tools[int(choice) - 1]
        else:  # 也允许直接输入工具名
            tool = next((t for t in tools if t.name == choice), None)
        if tool is None:
            print("无效选择：请输入列表中的编号，或 q 退出")
            continue

        arguments = {}
        cancelled = False
        for name, schema in _required_params(tool):
            label = PARAM_PROMPTS.get(name, f"请输入{name}")
            desc = (schema.get("description") or "").strip()
            if desc:
                label += f"（{desc}）"
            value = _ask(f"{label}: ")
            if not value:
                print("输入为空，已取消本次调用，返回菜单")
                cancelled = True
                break
            arguments[name] = _coerce(value, schema)
        if cancelled:
            continue

        label = tool.name
        if arguments:
            label += "(" + ", ".join(str(v) for v in arguments.values()) + ")"
        print(f"\n调用 {label} ...（rag_chat 含 LLM 生成，可能需要数十秒）")
        try:
            text = asyncio.run(_call_tool(url, tool.name, arguments))
        except Exception as exc:
            print(f"[调用失败] {exc}")
            print("（服务未启动或地址不对？确认 start_mcp.bat 已在运行、--url 正确）")
            continue

        # 输出格式化：候选文本可能含控制台代码页编不了的字符，替换而非崩溃
        try:
            sys.stdout.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
        print(f"\n===== {label} =====")
        try:
            print(json.dumps(json.loads(text), ensure_ascii=False, indent=2))
        except (json.JSONDecodeError, TypeError):
            print(text)
        print()


def main():
    parser = argparse.ArgumentParser(description="Interactive MCP client for Industrial RAG")
    parser.add_argument("--url", default=DEFAULT_URL,
                        help=f"MCP 服务地址（默认 {DEFAULT_URL}）")
    args = parser.parse_args()

    try:
        import mcp  # noqa: F401
    except ImportError:
        print("未安装 MCP SDK（pip install mcp==2.3.0）", file=sys.stderr)
        sys.exit(1)

    try:
        tools = sorted(asyncio.run(_list_tools(args.url)), key=lambda t: t.name)
        print(f"已连接 {args.url}，注册工具 {len(tools)} 个: {[t.name for t in tools]}")
    except Exception as exc:
        print(f"[警告] 拉取工具清单失败（{exc}）；使用内置工具列表，调用时仍会重试连接")
        tools = sorted(_FALLBACK_TOOLS, key=lambda t: t.name)

    try:
        _interactive(args.url, tools)
    except (KeyboardInterrupt, EOFError):
        print()
    print("已退出。")


if __name__ == "__main__":
    main()
