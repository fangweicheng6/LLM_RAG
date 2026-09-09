# RAG 多格式项目：文件作用与使用方法

本说明文档用于解释项目中**每一个 Python 文件 / 配置文件 / Parser 文件的作用，以及平时应该怎么使用**。

> 实际使用时，你通常只需要运行两个入口：
>
> ```bash
> python ingest.py
> python search.py "你的问题"
> ```
>
> 其他文件主要是内部模块，不需要单独运行。

---

# 1. 项目整体结构

```text
rag_multiformat_v2/
│
├── ingest.py
├── search.py
│
├── clean_chunk.py
├── embed_store.py
├── config.py
├── schema.py
├── requirements.txt
│
├── README.md
├── README_FILES.md
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

---

# 2. ingest.py

## 作用

`ingest.py` 是整个 RAG 系统的**一键入库入口**。

它本身不负责具体解析，而是依次调用：

```text
clean_chunk.py
    ↓
embed_store.py
```

最终完成：

```text
扫描知识库
→ 自动识别文件格式
→ 解析
→ 清洗
→ 切片
→ 生成标准 Chunk JSONL
→ Embedding
→ 写入 Chroma 向量库
```

## 平时怎么用

直接运行：

```bash
python ingest.py
```

通常你新增或修改知识文件后，再执行一次即可。

例如你往：

```text
D:\LLM_RAG\知识库\topology\buck
```

新增了：

```text
60_BUCK保护策略.md
```

然后运行：

```bash
python ingest.py
```

程序会重新处理并写入向量库。

## 是否需要修改

通常**不需要修改**。

后面增加 PDF、PPTX、JSON 等格式时，也尽量不要修改 `ingest.py`，而是增加新的 Parser。

---

# 3. search.py

## 作用

`search.py` 是 RAG 的**向量检索入口**。

执行流程：

```text
用户问题
→ BGE Embedding
→ 查询 Chroma
→ 计算向量相似度
→ 返回 Top-K Chunk
```

当前只是 Retriever，还没有接最终大模型回答。

## 使用方法

### 跨所有知识域检索

```bash
python search.py "BUCK电感如何计算？"
```

### 只查 topology

```bash
python search.py "BUCK电感如何计算？" --domain topology
```

### 只查 topology/buck

```bash
python search.py "BUCK电感如何计算？" --domain topology --subdomain buck
```


### 指定返回数量

```bash
python search.py "MOSFET耐压如何选择？" --top-k 10
```

## 是否需要修改

以后如果你加入：

```text
BM25
RRF
Reranker
Query Rewrite
Hybrid Search
```

主要就是扩展这个文件。

---

# 4. clean_chunk.py

## 作用

负责：

```text
原始文件
→ Parser
→ CanonicalBlock
→ 清洗
→ 分片
→ 标准 Chunk JSON
```

它是整个系统中**数据预处理的核心模块**。

但正常使用时，不需要直接执行它，因为 `ingest.py` 会自动调用。

## 主要做什么

### 1）扫描知识库

扫描：

```text
D:\LLM_RAG\知识库
```

找到所有支持的文件。

### 2）自动识别格式

例如：

```text
.md   → MarkdownParser
.pdf  → PDFParser
.docx → DOCXParser
.xlsx → XLSXParser
```

### 3）自动分域

例如：

```text
知识库/topology/buck/xxx.md
```

自动生成：

```json
{
  "domain": "topology",
  "subdomain": "buck"
}
```

例如：

```text
知识库/device/mosfet/xxx.pdf
```

自动生成：

```json
{
  "domain": "device",
  "subdomain": "mosfet"
}
```

### 4）统一清洗

主要处理：

```text
Unicode
换行
多余空格
HTML注释
连续空行
```

尽量保留：

```text
公式
单位
表格
Markdown结构
代码
技术符号
```

### 5）统一切片

当前属于：

```text
结构规则切片
+
字符长度控制
+
Overlap
```

不是 LLM 切片。

默认参数：

```python
MAX_CHARS = 900
OVERLAP_CHARS = 120
HARD_BLOCK_MAX_CHARS = 1600
```

### 6）输出 Chunk JSONL

默认输出：

```text
D:\LLM_RAG\processed\knowledge_chunks.jsonl
```

这个文件建议经常人工检查。

## 是否需要直接运行

一般不用。

如果你只想测试“解析和切片”，不想重新做 Embedding，可以单独运行：

```bash
python clean_chunk.py
```

---

# 5. embed_store.py

## 作用

负责：

```text
knowledge_chunks.jsonl
→ Embedding
→ Chroma
```

它完全不处理 PDF、Word、Excel 等原始文件。

也就是说它只认统一的 `Chunk Schema`。

## Embedding 模型

当前使用：

```text
BAAI/bge-large-zh-v1.5
```

用于把每个 Chunk 转换成 1024 维向量。

## 入库方式

按一级知识域自动创建 Chroma Collection：

```text
topology → rag_topology
device   → rag_device
pcb      → rag_pcb
```

以后新增：

```text
paper
experiment
```

会自动出现：

```text
rag_paper
rag_experiment
```

## 是否需要直接运行

平时不用。

因为：

```bash
python ingest.py
```

会自动调用它。

如果你只更换了 Embedding 模型，但不想重新清洗切片，可以单独运行：

```bash
python embed_store.py
```

这样直接读取已有：

```text
knowledge_chunks.jsonl
```

重新生成向量。

---

# 6. config.py

## 作用

集中保存整个项目的配置。

例如：

```python
KB_ROOT = Path(r"D:\LLM_RAG\知识库")
```

表示原始知识库目录。

```python
PROCESSED_DIR = Path(r"D:\LLM_RAG\processed")
```

表示清洗切片后的数据目录。

```python
CHROMA_DIR = Path(r"D:\LLM_RAG\vector_db\chroma")
```

表示向量数据库位置。

## 切片参数

```python
MAX_CHARS = 900
OVERLAP_CHARS = 120
HARD_BLOCK_MAX_CHARS = 1600
```

## Embedding 参数

```python
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
```

## 什么时候修改

常见修改：

### 修改知识库路径

```python
KB_ROOT = Path(r"E:\RAG\knowledge")
```

### 修改 Embedding 模型

```python
EMBEDDING_MODEL = "BAAI/bge-m3"
```

### 调整切片长度

```python
MAX_CHARS = 1200
```

一般建议优先修改 `config.py`，不要把参数散落到其他代码里。

---

# 7. schema.py

## 作用

定义整个 RAG 系统统一的数据结构。

这是项目里非常重要的文件。

主要有两个 Schema。

---

## CanonicalBlock

各种 Parser 最终都必须输出：

```python
CanonicalBlock
```

例如 PDF：

```text
PDF
→ text block
→ page=8
```

Word：

```text
Word
→ text/table block
→ section_path
```

Excel：

```text
Excel
→ table_row block
→ sheet_name
→ row_start / row_end
```

这样后面的清洗切片逻辑就不需要关心原始格式。

---

## Chunk

清洗和切片后统一形成：

```json
{
  "document_id": "...",
  "chunk_id": "...",
  "content": "...",

  "domain": "topology",
  "subdomain": "buck",
  "document_type": "markdown",

  "source_file": "...",
  "relative_path": "...",
  "source_format": "md",

  "title": "...",
  "section_path": "...",

  "page": null,
  "sheet_name": null,
  "row_start": null,
  "row_end": null,

  "tags": [],

  "chunk_index": 0,
  "char_count": 300,
  "content_hash": "..."
}
```

这个就是当前 RAG 的**标准 Chunk Schema**。

## 什么时候修改

以后如果你统一增加：

```text
version
language
project_id
manufacturer
device_model
permission_level
```

这类字段，应该优先在这里修改。

---

# 8. parsers/base.py

## 作用

定义所有 Parser 必须遵守的接口。

也就是：

```python
BaseParser
```

所有文件格式 Parser 都继承它。

例如：

```python
class PDFParser(BaseParser):
    ...
```

它规定 Parser 至少需要：

```text
extensions
document_type
parse()
```

## 为什么需要这个文件

这样新增格式时有统一规范。

例如新增 PowerPoint：

```python
class PPTXParser(BaseParser):
    extensions = (".pptx",)
    document_type = "powerpoint"

    def parse(self, path):
        ...
```

不会导致每种格式各写各的。

---

# 9. parsers/common.py

## 作用

存放 Parser 之间共享的辅助函数。

例如：

```text
文本编码兼容读取
表格行 → RAG文本
```

避免：

```text
CSVParser写一遍
TXTParser写一遍
MarkdownParser又写一遍
```

## 是否需要运行

不需要。

---

# 10. parsers/__init__.py

## 作用

Parser 注册中心。

程序通过这里知道：

```text
.md 由谁解析
.pdf 由谁解析
.docx 由谁解析
.xlsx 由谁解析
```

例如：

```python
PARSERS = [
    MarkdownParser(),
    TxtParser(),
    PDFParser(),
    DOCXParser(),
    XLSXParser(),
    CSVParser(),
    HTMLParser(),
]
```

然后自动建立：

```text
扩展名 → Parser
```

映射。

## 新增格式时需要修改这里

例如增加：

```text
pptx_parser.py
```

需要：

```python
from parsers.pptx_parser import PPTXParser
```

然后注册：

```python
PPTXParser(),
```

除此之外，主流程不用修改。

---

# 11. markdown_parser.py

## 支持格式

```text
.md
.markdown
```

## 主要功能

识别：

```text
#
##
###
```

这样的 Markdown Heading。

例如：

```markdown
# BUCK参数计算

## 电感设计

### 电感纹波
```

生成：

```text
section_path =
BUCK参数计算 > 电感设计 > 电感纹波
```

同时识别代码块：

```text
```python
...
```
```

避免在代码内部错误识别标题。

---

# 12. pdf_parser.py

## 支持格式

```text
.pdf
```

## 当前实现

使用：

```text
PyMuPDF
```

读取 PDF 文本层。

每页保留：

```json
"page": 8
```

方便以后答案追溯原始 PDF 页码。

## 当前适合

```text
普通电子版 PDF
Datasheet
普通论文
技术规范
```

## 当前不足

对于：

```text
扫描 PDF
复杂多栏论文
复杂 Datasheet 表格
公式
图片中的文字
```

第一版解析能力有限。

后续建议新增：

```text
MinerUParser
OCRParser
```

而不是推翻整个代码。

---

# 13. docx_parser.py

## 支持格式

```text
.docx
```

## 主要功能

读取 Word：

```text
标题
普通段落
表格
```

并尽量保持原始出现顺序。

Word Heading：

```text
Heading 1
Heading 2
Heading 3
```

会转换成：

```text
section_path
```

Word 表格会转换成 Markdown Table 后进入统一切片流程。

## 当前不支持

老式：

```text
.doc
```

如有需要，后面可以单独加 LegacyWordParser 或先批量转换为 `.docx`。

---

# 14. xlsx_parser.py

## 支持格式

```text
.xlsx
.xlsm
```

## 主要功能

使用：

```text
openpyxl
```

读取 Excel。

自动保留：

```text
sheet_name
row_start
row_end
```

第一行有效数据默认作为表头。

例如：

| Time | Voltage | Current |
|---|---:|---:|
| 0 | 48.2 | 5.01 |

会转成类似：

```text
Time: 0；Voltage: 48.2；Current: 5.01
```

然后再进入统一切片。

## 说明

当前是适合“表格型知识/实验记录”的通用版本。

如果以后实验数据特别大，例如几十万行时间序列，不建议所有原始数据直接 Embedding，需要再增加：

```text
统计摘要
实验批次聚合
原始数据引用
```

等策略。

---

# 15. csv_parser.py

## 支持格式

```text
.csv
```

## 功能

与 Excel 类似：

```text
表头
+
每一行数据
```

转换成适合 RAG 的文本。

例如：

```csv
Voltage,Current
48.2,5.01
```

转换为：

```text
Voltage: 48.2；Current: 5.01
```

同时保留原始行号。

---

# 16. txt_parser.py

## 支持格式

```text
.txt
```

## 功能

对纯文本做最基础解析。

后面统一交给：

```text
clean_chunk.py
```

完成清洗和切片。

---

# 17. html_parser.py

## 支持格式

```text
.html
.htm
```

## 主要功能

使用：

```text
BeautifulSoup
```

读取：

```text
标题
正文
列表
代码块
表格
```

会删除：

```text
script
style
noscript
```

等网页噪声。

HTML Heading 也会转换成：

```text
section_path
```

---

# 18. requirements.txt

## 作用

记录项目运行需要安装的 Python 依赖。

安装：

```bash
pip install -r requirements.txt
```

当前包括：

```text
chromadb
sentence-transformers
torch
numpy

PyMuPDF
python-docx
openpyxl
beautifulsoup4
lxml
```

以后增加新的 Parser 时，如果需要新的库，也应该加到这里。

例如新增 PPTX：

```text
python-pptx
```

---

# 19. README.md

## 作用

项目快速使用说明。

适合第一次打开项目时查看。

重点告诉你：

```text
如何安装
如何 ingest
如何 search
当前支持什么格式
如何扩展
```

---

# 20. README_FILES.md

就是当前这份文件。

它主要回答：

> 每个文件到底是干什么的？

适合后续维护代码或者团队成员接手时查看。

---

# 21. 实际日常使用流程

以后日常使用建议保持非常简单。

## 第一次安装

```bash
python -m venv .venv
```

激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 新增知识

例如：

```text
D:\LLM_RAG\知识库
├── topology
│   └── buck
│       └── 60_BUCK保护策略.md
│
├── device
│   └── mosfet
│       └── MOSFET选型指南.pdf
│
└── pcb
    └── layout
        └── PCB布局规范.docx
```

---

## 一键重新入库

```bash
python ingest.py
```

---

## 检索

```bash
python search.py "BUCK的MOSFET应该如何选择？"
```

---

# 22. 后续扩展格式的标准步骤

假设以后需要支持：

```text
.pptx
```

只做三步。

## 第一步

新增：

```text
parsers/pptx_parser.py
```

## 第二步

继承：

```python
BaseParser
```

并输出：

```python
CanonicalBlock
```

## 第三步

在：

```text
parsers/__init__.py
```

注册。

之后仍然只运行：

```bash
python ingest.py
```

不需要修改：

```text
search.py
embed_store.py
clean_chunk.py
```

---

# 23. 当前代码各层职责总结

```text
ingest.py
│
│   对外的一键入库入口
│
├── clean_chunk.py
│       │
│       ├── parsers/
│       │     └── 各种格式 → CanonicalBlock
│       │
│       ├── Cleaner
│       ├── Chunker
│       └── Chunk Schema
│
└── embed_store.py
        │
        ├── BGE Embedding
        └── Chroma Storage


search.py
│
├── Query Embedding
├── Metadata Filter
└── Vector Retrieval
```

可以简单记成：

```text
Parser     = 怎么读文件
Cleaner    = 怎么清洗
Chunker    = 怎么切
Schema     = 切完以后长什么样
Embedding  = 怎么变成向量
Storage    = 怎么存
Search     = 怎么查
```

这也是后面继续扩展整个 RAG 系统时最重要的代码边界。
