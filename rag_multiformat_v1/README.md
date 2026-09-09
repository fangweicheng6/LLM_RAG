# 多格式 RAG 数据管线 v2

这一版保留内部模块化结构，但用户实际只需要执行两个 Python 文件：

```text
python ingest.py
python search.py "你的问题"
```

也就是说：

```text
原始知识库
   ↓
python ingest.py
   ↓
自动完成：
多格式解析
→ 清洗
→ 切片
→ 标准 Chunk JSONL
→ Embedding
→ Chroma 入库

用户 Query
   ↓
python search.py "..."
   ↓
向量检索
```

---

## 1. 项目结构

```text
rag_multiformat_v2/
├── ingest.py             # 用户入口1：一键RAG入库
├── search.py             # 用户入口2：检索
│
├── clean_chunk.py        # 内部模块：解析、清洗、切片
├── embed_store.py        # 内部模块：Embedding、入库
├── config.py
├── schema.py
├── requirements.txt
│
└── parsers/
    ├── __init__.py
    ├── base.py
    ├── common.py
    ├── markdown_parser.py
    ├── pdf_parser.py
    ├── docx_parser.py
    ├── xlsx_parser.py
    ├── csv_parser.py
    ├── txt_parser.py
    └── html_parser.py
```

`clean_chunk.py` 和 `embed_store.py` 仍然保留，是为了代码职责分离和后续扩展，但平时不需要手工执行。

---

## 2. 当前支持格式

- Markdown：`.md` / `.markdown`
- PDF：`.pdf`
- Word：`.docx`
- Excel：`.xlsx` / `.xlsm`
- CSV：`.csv`
- TXT：`.txt`
- HTML：`.html` / `.htm`

---

## 3. 安装环境

推荐 Python 3.10 / 3.11。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -U pip
pip install -r requirements.txt
```

---

## 4. 配置知识库路径

默认：

```python
KB_ROOT = Path(r"D:\LLM_RAG\知识库")
```

在 `config.py` 修改。

---

## 5. 一键入库

只执行：

```powershell
python ingest.py
```

它会自动完成：

```text
扫描整个知识库
→ 根据后缀自动选择 Parser
→ 解析成 CanonicalBlock
→ 通用清洗
→ 分片
→ 自动确定 domain / subdomain
→ 输出 knowledge_chunks.jsonl
→ BAAI/bge-large-zh-v1.5 Embedding
→ 按 domain 自动写入 Chroma
```

中间文件：

```text
D:\LLM_RAG\processed\knowledge_chunks.jsonl
```

错误日志：

```text
D:\LLM_RAG\processed\processing_errors.jsonl
```

向量库：

```text
D:\LLM_RAG\vector_db\chroma
```

---

## 6. 检索

跨知识域：

```powershell
python search.py "BUCK电感如何计算？"
```

指定 domain：

```powershell
python search.py "BUCK电感如何计算？" --domain topology
```

指定 domain + subdomain：

```powershell
python search.py "BUCK电感如何计算？" --domain topology --subdomain buck
```

---

## 7. 为什么内部还保留多个 py

你要求的是“运行时只执行一个入库 Python + 一个检索 Python”，这版已经做到。

内部仍拆成：

```text
Parser
Cleaner/Chunker
Embedding/Storage
Search
```

是为了以后扩展 PDF、PPTX、Datasheet、PCB 文件时，不需要改一整个大脚本。

例如以后支持 PPTX，只新增：

```text
parsers/pptx_parser.py
```

再注册 Parser 即可，`ingest.py` 和 `search.py` 都不用改。


## 多媒体解析

PDF增强解析、图片OCR、PPTX、音频ASR、视频ASR+抽帧OCR的详细说明见 `README_MULTIMEDIA.md`。
