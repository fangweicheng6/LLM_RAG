# 配置说明

## 规则

项目运行时只读取 `configs/*.yaml`。`utils/config.py` 和 `utils/service_config.py` 已删除，代码中不再有旧的扁平常量或环境变量覆盖层。

`utils/config_loader.py` 会读取所有配置文件并合并为一个结构化对象 `settings`。每个叶子参数只能在一个 YAML 文件中定义；如果同一个参数被写入两个文件，程序启动时直接报错，避免“到底听谁的”变成隐式行为。

修改 YAML 后重启服务即可生效。长驻进程也可以调用 `reload_settings()` 重新加载，但已经创建的模型、向量客户端或服务实例不会自动重建，生产环境建议重启。

## 文件与参数

### `paths.yaml`

- `paths.kb_root`：原始知识库根目录。目录下按一级领域放置文件夹。
- `paths.data_dir`：索引运行目录。
- `paths.processed_dir`：清洗后的 JSONL 目录。
- `paths.chunk_jsonl`：无正文索引文件。
- `paths.error_jsonl`：解析失败记录。
- `paths.chroma_dir`：Chroma 向量库目录。
- `paths.source_store_dir`：受信任原文快照目录，必须与向量库分离。
- `paths.collection_prefix`：Chroma collection 前缀，例如 `rag_`。
- `paths.index_schema_version`：索引格式版本；修改后应重新建库。

相对路径相对于项目根目录解析；绝对路径可直接填写 Windows 或 Linux 路径。

### `ingestion.yaml`

- `ingestion.chunking.section_max_chars`：章节切片的最大字符数，过小会增加 chunk 数量，过大会增加上下文噪声。
- `ingestion.chunking.section_min_chars`：忽略的最小章节长度。
- `ingestion.pdf.ocr_min_text_chars`：PDF 提取文本少于该值时触发 OCR。
- `ingestion.pdf.ocr_zoom`：PDF OCR 渲染倍率，越高越清晰但越慢、越占内存。
- `ingestion.pdf.extract_tables`：是否提取 PDF 表格。
- `ingestion.ocr.enabled`：是否启用图片/扫描文档 OCR。
- `ingestion.ocr.min_score`：OCR 结果最低置信度。
- `ingestion.asr.model`：音视频 Whisper 模型名称。
- `ingestion.asr.language`：ASR 语言；`null` 表示自动识别。
- `ingestion.asr.beam_size`：ASR beam search 宽度，越大越慢但通常更稳定。
- `ingestion.asr.vad_filter`：是否启用语音活动检测。
- `ingestion.video.frame_ocr_enabled`：是否对视频帧执行 OCR。
- `ingestion.video.frame_ocr_interval_sec`：视频抽帧间隔（秒）。
- `ingestion.video.frame_ocr_dedup`：是否去重相邻帧的 OCR 文本。
- `ingestion.docling.max_tokens`：Docling chunk token 上限。
- `ingestion.docling.merge_peers`：是否合并同层级连续元素。
- `ingestion.embedding.batch_size`：Embedding 批大小，显存不足时调小。

### `model.yaml`

- `model.embedding.name`：Embedding 模型名称。
- `model.embedding.device`：期望设备（`cuda` 或 `cpu`）；运行时仍会检查实际可用设备。
- `model.embedding.query_instruction`：查询向量的指令前缀。
- `model.reranker.name`：CrossEncoder 模型名称。
- `model.reranker.top_k`：Reranker 保留的候选数。
- `model.llm.provider`：LLM provider 注册名。
- `model.llm.api_key`：API key；不要提交真实密钥到公共仓库。
- `model.llm.base_url`：兼容 OpenAI 的接口地址。
- `model.llm.model`：默认模型名（未在 `roles` 中单独指定的角色回退到此值）。
- `model.llm.roles`：按角色单独指定模型名，键为 `router` / `multi_hop` / `selector` / `generator` / `evaluation`；未配置的角色回退到 `model.llm.model`。其余参数（provider / base_url / temperature / disable_thinking）所有角色共享。
- `model.llm.timeout_sec`：单次 LLM 请求超时时间。
- `model.llm.max_retries`：请求失败重试次数。
- `model.llm.temperature`：生成随机性；事实型 RAG 通常保持 `0`。
- `model.generation.max_output_tokens`：最终答案最大输出 token 数。

### `rag.yaml`

- `rag.router.enabled`：是否调用 LLM 做知识域路由；关闭时检索全部可用领域。
- `rag.retrieval.top_k`：关闭 Context Selector 时的向量 Top-K。
- `rag.retrieval.candidate_pool_size_per_domain`：每个领域进入候选池的数量。
- `rag.retrieval.candidate_pool_max_chars`：发送给 Context Selector 的候选文本字符上限。
- `rag.retrieval.rerank.enabled`：是否启用重排序。
- `rag.retrieval.rerank.top_k`：重排序后保留数量。
- `rag.retrieval.multi_hop.enabled`：是否启用多跳查询拆解。
- `rag.retrieval.multi_hop.max_subqueries`：最多生成的子查询数。
- `rag.retrieval.multi_hop.cross_domain_enabled`：是否允许跨领域多跳规划。
- `rag.context.selector.enabled`：是否让 LLM 选择最小充分上下文。
- `rag.context.expander.enabled`：是否允许上下文补全。
- `rag.context.expander.max_rounds`：补全轮数上限。
- `rag.context.expander.max_requests`：每轮补全请求上限。
- `rag.context.expander.targeted_candidates`：定向补检候选数。
- `rag.context.expander.neighbor_before` / `neighbor_after`：邻近 chunk 前后扩展数量。

### `domains.yaml`

- `domains.allowed`：允许路由和索引使用的一级领域列表。
- `domains.descriptions.<domain>`：提供给 Router 的领域定义。新增领域时同时增加描述，并重建索引。

### `prompts.yaml`

包含关键词提取、领域路由、上下文选择和答案生成的所有 system/user 模板。模板中的 `{query}`、`{domain_text}`、`{candidate_text}` 等占位符由程序填充；如果要在模板中写字面量 JSON 花括号，使用 `{{` 和 `}}`。

### `service.yaml`

- `service.server.socket_host` / `socket_port`：Socket 服务监听地址和端口。
- `service.server.http_host` / `http_port`：HTTP 服务监听地址和端口。
- `service.observability.langfuse.enabled`：是否启用 Langfuse。
- `service.observability.prometheus.enabled`：是否启用 Prometheus 指标。
- `service.runtime.preload_models`：启动时是否预加载模型。
- `service.runtime.preload_vector_db`：启动时是否预加载向量库。

### `evaluation.yaml` 与 `dataset_generation.yaml`

`evaluation.*` 控制评估集路径、模型、超时、重试、指标权重和 benchmark 范围；`dataset_generation.*` 控制评估数据集总量、问题类型比例、意图比例及校验策略。修改比例时请确保总和为 `1.0`。

## 常用修改

关闭所有增强模块：在 `rag.yaml` 中将 `router.enabled`、`retrieval.rerank.enabled`、`retrieval.multi_hop.enabled`、`context.selector.enabled` 和 `context.expander.enabled` 设为 `false`。

切换模型：只修改 `model.yaml` 的 `model.embedding.*` 或 `model.llm.*`，然后重新建立向量索引（更换 Embedding 模型时必须重建）。

修改端口：只修改 `service.yaml` 的端口并重启对应入口。
