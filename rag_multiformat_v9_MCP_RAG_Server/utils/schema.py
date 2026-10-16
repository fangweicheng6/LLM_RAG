# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, asdict, field
import json
from utils.config_loader import settings
from typing import Optional, List, Dict, Any


@dataclass
class CanonicalBlock:
    """
    所有原始格式都先统一解析成 CanonicalBlock。

    例：
    - Markdown / Word / HTML -> text / table / code
    - PDF -> text / table / ocr
    - Excel / CSV -> table_row
    - PPTX -> slide_text / table / image_ocr
    - Image -> image_ocr
    - Audio / Video -> transcript
    - Video frame -> video_frame_ocr
    """

    content_type: str
    content: str

    # 文档结构定位
    section_path: str = ""

    # PDF
    page: Optional[int] = None
    page_end: Optional[int] = None

    # Excel / CSV
    sheet_name: Optional[str] = None
    row_start: Optional[int] = None
    row_end: Optional[int] = None

    # PowerPoint
    slide_number: Optional[int] = None

    # 图片（例如 PPT 中第几张图片）
    image_index: Optional[int] = None

    # 音频 / 视频时间定位
    media_start_sec: Optional[float] = None
    media_end_sec: Optional[float] = None


@dataclass
class Chunk:
    """
    RAG 标准 Chunk Schema。
    """
    document_id: str
    chunk_id: str

    content: str

    domain: str
    subdomain: str
    document_type: str

    source_file: str
    relative_path: str
    source_format: str

    title: str
    section_path: str

    page: Optional[int]
    sheet_name: Optional[str]
    row_start: Optional[int]
    row_end: Optional[int]

    tags: List[str]

    chunk_index: int
    char_count: int
    content_hash: str

    # 章节可能跨 PDF 多页；给默认值以兼容旧 JSONL。
    page_end: Optional[int] = None

    # 章节级稳定定位。
    # 同一文档、同一 section_path 的多个超长分块共享 section_id。
    section_id: Optional[str] = None
    section_chunk_index: Optional[int] = None
    section_chunk_count: Optional[int] = None

    # v3 多媒体扩展字段。给默认值，兼容旧 JSONL。
    content_type: str = "text"
    slide_number: Optional[int] = None
    image_index: Optional[int] = None
    media_start_sec: Optional[float] = None
    media_end_sec: Optional[float] = None

    # 仅定位数据，正文只在内存和外部原文目录中存在。
    source_ref: str = ""
    source_unit: str = ""
    source_file_hash: str = ""
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    column_start: Optional[int] = None
    column_end: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    line_basis: str = "parsed_text_v1"
    keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """仅用于内存对象；落盘必须使用 to_index_dict。"""
        return asdict(self)

    def chroma_metadata(self) -> Dict[str, Any]:
        """显式白名单；title/section_path/tags/content 永远不写入向量库。"""
        names = (
            "document_id", "domain", "subdomain", "document_type",
            "source_file", "relative_path", "source_format", "section_id",
            "section_chunk_index", "section_chunk_count", "content_type",
            "page", "page_end", "sheet_name", "row_start", "row_end",
            "slide_number", "image_index", "media_start_sec", "media_end_sec",
            "chunk_index", "char_count", "content_hash", "source_ref",
            "source_unit", "source_file_hash", "line_start", "line_end",
            "column_start", "column_end", "char_start", "char_end", "line_basis",
        )
        meta = {name: getattr(self, name) for name in names if getattr(self, name) is not None}
        meta["index_schema_version"] = settings.paths.index_schema_version
        meta["keywords"] = json.dumps(self.keywords, ensure_ascii=False)
        return meta

    def to_index_dict(self) -> Dict[str, Any]:
        return {"chunk_id": self.chunk_id, **self.chroma_metadata()}
