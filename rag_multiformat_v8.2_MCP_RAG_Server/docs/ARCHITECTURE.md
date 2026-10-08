# 项目结构与程序入口

## 请求链路

```text
server/start_server.py
  ├─ deployment/rag_server.py        Socket
  └─ deployment/api_server.py        HTTP
          ↓
server/runtime/service_runtime.py
          ↓
server/runtime/component_loader.py
          ↓
core/engine/rag_engine.py
          ↓
pipeline/retrieval/retriever.py
  ├─ pipeline/routing/router.py      领域路由与关键词
  ├─ pipeline/retrieval/multi_hop.py 多跳拆解
  ├─ Chroma 向量召回
  ├─ pipeline/retrieval/reranker.py  重排序
  ├─ pipeline/context/selector.py    上下文选择
  ├─ pipeline/context/expander.py    上下文补全
  └─ pipeline/generation/generator.py答案生成
```

## 目录职责

- `configs/`：唯一运行配置来源，按路径、摄取、模型、RAG、领域、Prompt、服务和评估分类。
- `utils/config_loader.py`：读取 YAML、校验重复配置、解析相对路径，暴露结构化 `settings`。
- `parsers/`：PDF、Office、HTML、表格、图片、音视频等格式解析。
- `ingestion/`：知识库扫描、清洗切片、原文快照、Embedding 和 Chroma 入库。
- `pipeline/`：在线 RAG 路由、检索、上下文处理和回答生成。
- `llm/`：LLM 抽象、provider 注册和 OpenAI 兼容实现。
- `server/` 与 `deployment/`：服务生命周期、Socket/HTTP 入口和运行时组件装配。
- `evaluation/`：评估集生成、四类 benchmark、Ragas 适配。
- `observability/`：Langfuse、Prometheus 和 OpenTelemetry 相关能力。
- `tests/`：索引安全、检索和评估测试。

## 入口

### 建立索引

在项目根目录运行：

```powershell
python -m ingestion.ingest
```

该命令依次执行解析/清洗切片和 Embedding/Chroma 入库，路径来自 `configs/paths.yaml`。

### 启动 Socket 服务

```powershell
python -m server.start_server --mode socket
```

监听地址和端口来自 `configs/service.yaml` 的 `service.server.socket_*`。

### 启动 HTTP 服务

```powershell
python -m server.start_server --mode http
```

HTTP 入口为 `deployment.api_server:app`，端口来自 `service.server.http_*`。

### 运行评估

```powershell
python -m evaluation.run_benchmark_evaluation --type all
```

评估数据和输出目录由 `configs/evaluation.yaml` 管理。

### 索引审计

```powershell
python -m utils.index_audit
```

检查 Chroma 是否含正文、metadata 是否符合白名单以及原文存储目录是否隔离。

## 配置生效边界

每次进程启动时加载一次 YAML。模型、Chroma client 和服务对象在启动期间缓存；修改 YAML 后应重启进程。`utils.config_loader.reload_settings()` 只刷新配置对象，适合测试和管理脚本，不会替换已经创建的模型实例。

