# -*- coding: utf-8 -*-
from pathlib import Path

# ============================================================
# 路径
# ============================================================

KB_ROOT = Path(r"D:\LLM_RAG\知识库")

PROCESSED_DIR = Path(r"D:\LLM_RAG\results\processed")
CHUNK_JSONL = PROCESSED_DIR / "knowledge_chunks.jsonl"
ERROR_JSONL = PROCESSED_DIR / "processing_errors.jsonl"

CHROMA_DIR = Path(r"D:\LLM_RAG\results\vector_db\chroma")

# ============================================================
# 通用切片
# ============================================================

MAX_CHARS = 900
OVERLAP_CHARS = 120
HARD_BLOCK_MAX_CHARS = 1600

# ============================================================
# PDF
# ============================================================

# 文本层少于该字符数时，认为可能是扫描页，触发 OCR fallback。
PDF_OCR_MIN_TEXT_CHARS = 40

# OCR 渲染倍率。2.0 对多数技术 PDF 足够。
PDF_OCR_ZOOM = 2.0

# 是否尝试 PyMuPDF 的表格识别。
PDF_EXTRACT_TABLES = True

# ============================================================
# OCR：图片 / 扫描 PDF / PPT 图片 / 视频帧
# ============================================================

OCR_ENABLED = True

# RapidOCR 置信度下限。
OCR_MIN_SCORE = 0.45

# ============================================================
# 音频 / 视频 ASR
# ============================================================

# faster-whisper 模型，可改为 tiny/base/small/medium/large-v3。
# 第一版默认 small，兼顾资源与效果。
WHISPER_MODEL = "small"

# None = 自动识别语言；如果音频固定中文可改成 "zh"。
WHISPER_LANGUAGE = None

WHISPER_BEAM_SIZE = 5
WHISPER_VAD_FILTER = True

# ============================================================
# 视频帧 OCR
# ============================================================

# 视频除了语音转写外，是否抽帧做 OCR。
VIDEO_FRAME_OCR_ENABLED = True

# 每隔多少秒抽一帧做 OCR。
VIDEO_FRAME_OCR_INTERVAL_SEC = 15.0

# 相邻抽帧 OCR 完全相同则去重。
VIDEO_FRAME_OCR_DEDUP = True

# ============================================================
# Embedding
# ============================================================

# EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
EMBEDDING_MODEL = "BAAI/bge-base-zh-v1.5"

QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
EMBED_BATCH_SIZE = 16

# 一个一级 domain 对应一个 Chroma collection。
COLLECTION_PREFIX = "rag_"
