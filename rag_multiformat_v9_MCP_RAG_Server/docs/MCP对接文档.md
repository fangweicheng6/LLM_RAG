# MCP 对接文档（交接版）

> 面向对接方工程师。本服务通过 **MCP（Model Context Protocol）streamable-http**
> 对外提供知识库检索能力：对接方拿到相关 chunk 后，用自家 LLM 生成最终回答。
>
> 对应关系一句话：**本服务不生成答案，只提供带出处定位的知识块。**

- 服务名称：`industrial-rag`（版本 8.2.0）
- 默认端点：`http://127.0.0.1:8100/mcp`
- 协议：MCP streamable-http（HTTP POST + 可选 SSE 长连接，由 MCP 官方 SDK 封装）

---

## 1. 服务端部署（本侧）

### 1.1 启动（任选其一，等价）

```powershell
# 在项目根目录
python -m server.start_server --mode mcp-http        # 推荐
python -m deployment.mcp_server --transport streamable-http
start_mcp.bat                                        # 一键启动（默认 streamable-http）
```

启动成功后 stderr 会打印：`MCP streamable-http: http://127.0.0.1:8100/mcp`。

### 1.2 地址与端口

配置在 `configs/service.yaml`：

```json
"mcp": {
  "http_host": "127.0.0.1",        // 跨机器对接改为 "0.0.0.0" 或指定网卡 IP
  "http_port": 8100,
  "streamable_http_path": "/mcp",
  "warmup": true,                   // 启动即后台预热模型
  "expose_admin_tools": false,      // 是否暴露 rag_ingest / rag_index_audit
  "call_lock_timeout": 120
}
```

- **跨机器对接**：`http_host` 改成 `"0.0.0.0"`，重启服务，并在 Windows 防火墙放行
  **8100/TCP**（入站）。
- 修改配置后需要**重启服务进程**。

### 1.3 安全说明（重要）

- 当前 MCP 端点**无鉴权、无 TLS**。请部署在可信内网；
  若必须跨不可信网络，请在网关/反代层加认证与加密。
- 建议限制来源 IP（防火墙白名单只放对接方主机）。

### 1.4 运行环境

- 需要本项目的 Python 环境（`D:\...\rag\Scripts\python.exe`，已装好依赖）。
- 环境变量 `LLM_API_KEY`：只有"服务端仍然调用 LLM 的环节"才需要。
  当前交接形态（见 §6）已关闭回答生成；若同时关闭 Context Selector 与
  multi-hop，则整个服务零 LLM 调用、不需要该变量。

---

## 2. 对接方如何连接

### 2.1 Python（官方 SDK `mcp`，已实测）

```python
import asyncio, json
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

URL = "http://<服务端IP>:8100/mcp"     # 同机时用 127.0.0.1

async def main():
    async with streamable_http_client(URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print([t.name for t in tools.tools])

            # 首次调用会触发服务端加载模型（1~2 分钟），超时给足
            result = await session.call_tool(
                "rag_search",
                {"query": "BUCK 电感如何计算"},
                read_timeout_seconds=600,
            )
            data = json.loads(result.content[0].text)
            for c in data["candidates"][:5]:
                print(c["chunk_id"], c["score"], c["source_file"], c["location"])

asyncio.run(main())
```

> 安装 SDK：`pip install mcp`（服务端实测版本 2.3.0；1.x 版本的客户端 API
> 差异为 `from mcp.client.streamable_http import streamablehttp_client`
> 且返回三元组，按所装版本调整）。

### 2.2 其他语言

MCP 官方提供多语言 SDK（TypeScript / Java / Kotlin / C# / Go / Rust 等），
或使用任何兼容 MCP 的客户端框架。核心交互一致：

`initialize` → `tools/list` → `tools/call`。

### 2.3 会话模型

- 一次连接内可连续调用任意多次工具；服务端一个连接一个会话。
- 客户端断开重连即可，无状态丢失（检索无会话状态）。
- 首次 `tools/call` 之前若有调用排队，可能等到服务端预热完成（见 §5 超时说明）。

---

## 3. 工具契约

### 3.1 `rag_search` —— 对接方主要使用的工具

**入参**：`{"query": "<检索问题，中文/英文均可>"}`

**返回**（JSON 字符串，即 `content[0].text`，需自行 `json.loads`）：

```jsonc
{
  "query": "BUCK 电感如何计算",
  "router_domains": ["topology"],
  "domains": ["topology"],
  "count": 2,
  "candidates": [
    {
      "chunk_id": "DOC_3A4D74DD2A9D_0001_6C440493",
      "collection": "rag_topology",
      "score": 0.93,                    // 重排分（无重排时为向量相似度），已按相关度排好序
      "source_file": "topology/buck/04_参数计算规则.md",
      "domain": "topology",
      "location": "parsed_unit=1 lines=1-40 columns=1-25",   // 出处定位（页/行/列）
      "text": "chunk 全文（最多 2000 字符，超出截断）"
    }
  ]
}
```

**使用建议**：直接按返回顺序取前 5~8 条作为 LLM 上下文；
引用标注用 `chunk_id` + `source_file` + `location`。

**截断说明**（防止撑爆客户端上下文，属设计行为）：
最多返回 **20 个候选**，每个 `text` 最多 **2000 字符**。
需要更长正文时：调大服务端 `deployment/mcp_server.py` 中的
`_SEARCH_TOP_K` / `_TEXT_MAX_CHARS`，或改用 HTTP 接口 `POST /search`
（返回完整 `document` 字段）。

### 3.2 `rag_chat` —— 完整问答工具（本部署已关闭服务端生成）

**入参**：`{"question": "<用户问题>"}`

**当前部署（`rag.generation.enabled=false`）返回**：

```jsonc
{
  "answer": null,                     // 服务端不生成回答（显式 null，不是错误）
  "generation_enabled": false,        // 本响应未经回答生成阶段
  "context_sufficient": true,         // 上下文选择器的判断：信息是否足够
  "missing_information": [],          // 按选择器判断仍缺的信息点
  "router_domains": ["topology"],
  "domains": ["topology"],
  "selection_reason": "…",            // 选择器给出这些块的理由
  "sources": [],                      // 无生成则无引用；需要引用自行用 selected_contexts 拼
  "selected_contexts": [
    {
      "chunk_id": "DOC_3A4D74DD2A9D_0001_6C440493",
      "collection": "rag_topology",
      "source_file": "topology/buck/04_参数计算规则.md",
      "domain": "topology",
      "score": 0.93,
      "text": "…（最多 8 条 × 每条约 2000 字符）"
    }
  ]
}
```

> 与 `rag_search` 的区别：`rag_chat` 会先经过 LLM 上下文选择器
> （更准，但多一次 LLM 调用，且返回条数更少、更精）；`rag_search`
> 只做检索+重排（更快、更多候选）。对接方通常用 `rag_search`。

**若未来把服务端生成打开**（`rag.generation.enabled=true`），
`answer` 会变为字符串答案、`generation_enabled=true`、`sources` 为引用列表；
对接方代码按 `generation_enabled` 字段分支即可兼容两种形态。

### 3.3 `rag_health`

无入参。返回 `{"status": "ok", "components": ["router", "retriever", "generator", "observability"]}`。
用于探活。

### 3.4 `rag_list_sources`

无入参。返回各知识域 collection 与块数量：

```jsonc
{"collection_prefix": "rag_", "collections": [{"collection": "rag_topology", "chunks": 2352}]}
```

### 3.5 管理工具（默认不暴露）

`rag_ingest`（更新知识库，`incremental=true` 时只重解析新增/变化的文件并清理已删文件的 chunk）、`rag_index_audit`（索引审计）默认关闭；
需服务端在 `configs/service.yaml` 设 `service.mcp.expose_admin_tools=true`
（或环境变量 `RAG_MCP_ADMIN_TOOLS=1`）并重启。对接方一般不需要。

### 3.6 错误契约

- 工具执行异常 → MCP **工具错误**（`isError=true`，content 为 `"<动作> 失败: <原因>"` 文本）。
  对接方应对 `call_tool` 的 error 分支做处理并在日志里记录文本。
- 已知排队超时：服务端预热/上一调用未完成时，工具调用可能在
  `call_lock_timeout`（默认 120 秒）后返回"排队超时，请稍后重试"类错误 —— **可重试**。
- 空 `query`/`question` 会直接返回参数错误。

---

## 4. 对接方完整接入示例（含 LLM）

```python
import asyncio, json
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

URL = "http://<服务端IP>:8100/mcp"

async def retrieve(question: str, top_k: int = 6) -> list[dict]:
    async with streamable_http_client(URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            r = await session.call_tool(
                "rag_search", {"query": question}, read_timeout_seconds=600
            )
            if r.isError:
                raise RuntimeError(r.content[0].text)
            data = json.loads(r.content[0].text)
            return data["candidates"][:top_k]

def build_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] 来源: {c['source_file']} | {c['location']}\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    )

async def main():
    q = "BUCK 电感如何计算？"
    chunks = await retrieve(q)
    if not chunks:
        print("知识库中没有相关内容")
        return
    prompt = (
        "仅依据以下知识库片段回答问题，逐条标注引用编号；"
        "片段不足以回答时明确说明。\n\n"
        + build_context(chunks)
        + f"\n\n问题：{q}"
    )
    # answer = your_llm_chat(prompt)      # ← 接你自己的 LLM

asyncio.run(main())
```

---

## 5. 性能与超时提示

| 事项 | 说明 |
| --- | --- |
| 首次调用 | 服务端懒加载嵌入模型/重排器/向量库（约 1~2 分钟；同机 GPU 更快）。之后单次查询通常 1~5 秒 |
| 预热 | 服务端启动即后台预热（`service.mcp.warmup=true`；环境变量 `RAG_MCP_WARMUP=0` 可关） |
| 客户端超时 | 建议 `read_timeout_seconds >= 300`（官方 SDK 默认超时较短，务必显式设置） |
| 并发 | 服务端串行执行工具调用（一把调用锁），高并发请对接方侧排队；超时见 §3.6 |

---

## 6. 部署形态与开关（服务端配置）

`configs/rag.yaml`：

```jsonc
{
  "rag": {
    "router":    { "enabled": false },                     // 默认关：全域检索
    "generation":{ "enabled": false },                     // 当前默认关：不生成回答，只给 chunk
    "context":   { "selector": { "enabled": true } },      // 上下文选择器（一次 LLM 调用）
    "retrieval": { "multi_hop": { "enabled": true } }      // 多跳扩展（一次 LLM 调用）
  }
}
```

三种典型形态：

| 形态 | generation | selector | multi_hop | 特点 |
| --- | --- | --- | --- | --- |
| **交接默认** | false | true | true | chunk 由 LLM 选择器精选；服务端仍调用 LLM（需要 LLM_API_KEY），但**不生成回答** |
| 零 LLM 纯检索 | false | false | false | 服务端不发任何 LLM 请求，**不需要 LLM_API_KEY**；chunk 由向量+词法+重排给出 |
| 内置问答 | true | true | true | `rag_chat` 直接返回生成好的答案（含引用），与旧版本行为一致 |

> 改配置后重启服务生效。

---

## 7. 交接验收清单

- [ ] 服务端启动，stderr 出现 `MCP streamable-http: http://…:8100/mcp`
- [ ] 跨机器时：对接方主机能连通 8100（`Test-NetConnection <IP> -Port 8100`）
- [ ] 对接方 `initialize` 成功，`tools/list` 返回 4 个工具
      （`rag_chat` / `rag_search` / `rag_health` / `rag_list_sources`）
- [ ] `rag_health` 返回 `status: ok`
- [ ] `rag_search("一个业务样例问题")` 返回 `count > 0`，`text` 非空
- [ ] `rag_chat(...)` 返回 `answer: null` 且 `generation_enabled: false`（当前形态）
- [ ] 对接方端到端跑通：检索 → 自家 LLM → 带引用的回答

---

## 8. 常见问题

**Q：连接被拒绝 / 超时？**
检查 `http_host` 是否为 `127.0.0.1`（跨机器必须改 `0.0.0.0`）、防火墙是否放行 8100/TCP、
服务端是否在运行。

**Q：第一次调用很慢？**
模型懒加载 + 预热，属正常；客户端超时设置 ≥300 秒。

**Q：`answer` 是 `null` 是坏了吗？**
不是。当前部署关闭了服务端生成（`generation_enabled: false`），
`selected_contexts` / `rag_search.candidates` 就是你要的数据。

**Q：`text` 为什么被截断？**
防上下文爆炸的设计上限（20×2000 / 8×2000），见 §3.1；可调或走 HTTP `/search`。

**Q：能不能不用 MCP，直接 HTTP？**
可以，同一服务同时开 HTTP 接口（`POST /chat`、`POST /search`，端口 8000），
返回同一份底层数据、无截断；详见 `docs/对接说明.md`。
本交接约定以 MCP streamable-http 为准。
