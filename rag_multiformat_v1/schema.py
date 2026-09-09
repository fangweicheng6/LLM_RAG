# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, asdict
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

    # v3 多媒体扩展字段。给默认值，兼容旧 JSONL。
    content_type: str = "text"
    slide_number: Optional[int] = None
    image_index: Optional[int] = None
    media_start_sec: Optional[float] = None
    media_end_sec: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def chroma_metadata(self) -> Dict[str, Any]:
        """
        Chroma metadata 使用标量值。
        tags 是 list，因此保留在标准 JSONL，但第一阶段不直接写入 Chroma。
        None 字段不写入。
        """
        meta = {
            "document_id": self.document_id,
            "domain": self.domain,
            "subdomain": self.subdomain,
            "document_type": self.document_type,
            "source_file": self.source_file,
            "relative_path": self.relative_path,
            "source_format": self.source_format,
            "title": self.title,
            "section_path": self.section_path,
            "content_type": self.content_type,

            "page": self.page,
            "sheet_name": self.sheet_name,
            "row_start": self.row_start,
            "row_end": self.row_end,

            "slide_number": self.slide_number,
            "image_index": self.image_index,

            "media_start_sec": self.media_start_sec,
            "media_end_sec": self.media_end_sec,

            "chunk_index": self.chunk_index,
            "char_count": self.char_count,
            "content_hash": self.content_hash,
        }
        return {k: v for k, v in meta.items() if v is not None}
