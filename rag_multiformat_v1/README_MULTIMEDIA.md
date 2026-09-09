# 多格式 / 多媒体 Parser 设计说明

本版本保持两个用户入口不变：

```bash
python ingest.py
python search.py "你的问题"
```

`ingest.py` 内部会自动识别文件格式并选择对应 Parser。

这次主要升级了：

PDF：从“整页纯文本”升级为“标题层级 + 表格 + 扫描页 OCR fallback”
Office：支持 DOCX、XLSX/XLSM、PPTX
图片：支持 PNG/JPG/TIFF/WebP 等 OCR
音频：支持 MP3/WAV/M4A/FLAC 等 ASR
视频：支持 MP4/MKV/MOV 等，做“语音转写 + 定时抽帧 OCR”
Parser 注册：改成自动发现，不再手工维护扩展名映射



---

# 1. 当前支持格式

## 文档

```text
.md / .markdown
.txt
.html / .htm
.pdf
.docx
.xlsx / .xlsm
.csv
.pptx
```

## 图片

```text
.png
.jpg / .jpeg
.bmp
.tif / .tiff
.webp
```

## 音频

```text
.mp3
.wav
.m4a
.flac
.aac
.ogg
.wma
.opus
```

## 视频

```text
.mp4
.mov
.avi
.mkv
.webm
.m4v
.wmv
.flv
```

老式 `.doc / .xls / .ppt` 当前不直接解析，推荐先转换成 Office Open XML：
`.docx / .xlsx / .pptx`。

---

# 2. 所有格式统一成什么

无论源文件是什么，Parser 都先输出：

```python
CanonicalBlock
```

统一字段包括：

```text
content_type
content
section_path

page

sheet_name
row_start
row_end

slide_number
image_index

media_start_sec
media_end_sec
```

不同格式只填写自己有意义的定位字段。

---

# 3. PDF Parser v3 做了什么优化

旧版 PDF：

```text
一页 PDF
→ page.get_text("text")
→ 整页纯文本
```

新版：

```text
PDF
→ 读取 text dict / span / font / bbox
→ 字体大小与粗体启发式标题识别
→ section_path
→ 表格检测
→ 普通文本块
→ 扫描页 OCR fallback
→ CanonicalBlock
```

## 3.1 标题识别

根据：

```text
字体大小
粗体
文字长度
```

判断标题，并维护：

```text
一级标题 > 二级标题 > 三级标题
```

从而让 PDF 也能产生类似 Markdown 的 `section_path`。

## 3.2 表格

使用 PyMuPDF：

```python
page.find_tables()
```

识别表格后转成 Markdown Table。

同时跳过与表格 bbox 大面积重叠的普通文本，减少重复。

## 3.3 扫描 PDF

如果某页文本层字符过少：

```text
字符数 < PDF_OCR_MIN_TEXT_CHARS
```

则：

```text
PDF page
→ 渲染图片
→ RapidOCR
→ content_type = ocr
```

---

# 4. 图片

图片 Parser：

```text
PNG / JPG / TIFF / ...
→ RapidOCR
→ image_ocr
```

输出例如：

```json
{
  "content_type": "image_ocr",
  "content": "MOSFET Vds 60V ...",
  "image_index": 1
}
```

第一版主要解决“图片中的文字进入 RAG”。

以后如果需要理解没有文字的电路图 / PCB 图，可再增加：
Vision Caption / 图像模型 Parser。

---

# 5. Word

DOCX：

```text
Heading
Paragraph
Table
```

转成：

```text
text / table CanonicalBlock
```

Heading 转换为 `section_path`。

---

# 6. Excel / CSV

表格：

```text
表头 + 行
```

转成：

```text
字段名: 值；字段名: 值
```

例如：

```text
Voltage: 48.2；Current: 5.01
```

同时保留：

```text
sheet_name
row_start
row_end
```

---

# 7. PowerPoint

PPTX 支持：

```text
文本框
表格
图片 OCR
```

每个 Block 保留：

```text
slide_number
```

例如：

```json
{
  "content_type": "slide_text",
  "slide_number": 5,
  "section_path": "Slide 5 > BUCK控制策略"
}
```

PPT 中图片里的文字会通过 RapidOCR 生成 `image_ocr`。

---

# 8. 音频

音频使用：

```text
faster-whisper
```

进行 ASR。

输出不是一整篇转写，而是带时间戳的 segment：

```json
{
  "content_type": "transcript",
  "content": "这里介绍BUCK电感计算方法",
  "media_start_sec": 12.4,
  "media_end_sec": 18.7
}
```

后续 Chunker 会合并相邻转写片段，并保留整个 Chunk 的开始和结束时间。

---

# 9. 视频

视频做两条解析链：

```text
视频
├─ 音轨 → faster-whisper → transcript
└─ 画面 → 定时抽帧 → RapidOCR → video_frame_ocr
```

默认：

```python
VIDEO_FRAME_OCR_INTERVAL_SEC = 15.0
```

即每 15 秒抽一帧。

这特别适合：

```text
课程录像
PPT讲解视频
实验视频
屏幕录制
```

因为既能检索“说了什么”，也能检索“屏幕上出现了什么文字”。

---

# 10. Parser 注册现在也是自动的

不再手工维护：

```python
".pdf" -> PDFParser
".docx" -> DOCXParser
...
```

`parsers/__init__.py` 会自动扫描所有：

```text
*_parser.py
```

并自动发现 `BaseParser` 子类。

所以以后支持新的格式，例如：

```text
json_parser.py
mat_parser.py
pcbdoc_parser.py
```

只要：

```python
class XxxParser(BaseParser):
    extensions = (".xxx",)

    def parse(self, path):
        return [...]
```

就会自动注册。

无需修改：

```text
ingest.py
clean_chunk.py
embed_store.py
search.py
parsers/__init__.py
```

---

# 11. 推荐运行方式

安装依赖：

```bash
pip install -r requirements.txt
```

一键入库：

```bash
python ingest.py
```

检索：

```bash
python search.py "BUCK电感如何计算？"
```

指定知识域：

```bash
python search.py "MOSFET耐压如何选择？" --domain device
```

---

# 12. 注意事项

## PDF

增强版已经比整页纯文本更好，但复杂双栏论文、复杂 Datasheet 表格仍可能出现阅读顺序问题。

如果后续对 PDF 要求更高，推荐新增独立：

```text
MinerUParser
```

让 MinerU 输出统一 `CanonicalBlock`。

主流程无需修改。

## OCR

RapidOCR 主要用于文字识别。

对于：

```text
电路图语义
PCB图片结构
器件外观
波形图含义
```

还需要后续增加视觉模型，而不是只依赖 OCR。

## 音视频

第一次使用 `faster-whisper` 会下载对应 Whisper 模型。

默认：

```python
WHISPER_MODEL = "small"
```

如果机器性能较好，可以改：

```python
WHISPER_MODEL = "medium"
```

或：

```python
WHISPER_MODEL = "large-v3"
```

以提高转写准确率。
