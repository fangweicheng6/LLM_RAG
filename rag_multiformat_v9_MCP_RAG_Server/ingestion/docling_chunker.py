# -*- coding: utf-8 -*-
"""
Docling parser + HybridChunker adapter for RAG ingestion.

保持项目原有 Chunk / source_store 数据契约，只替换文档解析与切片层：
原：parser -> CanonicalBlock -> section split -> Chunk
新：Docling -> DoclingDocument -> HybridChunker -> Chunk

重要：项目采用“索引不保存正文”的设计。每个 Docling chunk 都必须拥有可回溯的
source_unit / char_start / char_end，并在 source_store 中保存对应解析文本，
否则 Embedding 阶段的 SourceResolver 无法 hydrate 正文。
"""

from __future__ import annotations

from collections import Counter
import hashlib
import logging
from pathlib import Path
from typing import Any, Iterable

from utils.config_loader import settings
from utils.schema import Chunk
from ingestion.source_store import digest, extract_keywords, save_snapshot
from ingestion.chunk_clean import clean_chunk_text, content_fingerprint, is_junk_chunk


def sha1_short(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n].upper()


def stable_document_id(relative_path: str) -> str:
    return f"DOC_{sha1_short(relative_path)}"


def stable_chunk_id(document_id: str, chunk_index: int, content: str) -> str:
    return f"{document_id}_{chunk_index:04d}_{sha1_short(content, 8)}"


def _docling_import():
    try:
        from docling.chunking import HybridChunker
        from docling.document_converter import DocumentConverter
        from docling_core.transforms.chunker.tokenizer.huggingface import (
            HuggingFaceTokenizer,
        )
        from transformers import AutoTokenizer

        return (
            DocumentConverter,
            HybridChunker,
            HuggingFaceTokenizer,
            AutoTokenizer,
        )
    except ImportError as exc:
        raise RuntimeError(
            "当前环境未安装 docling / docling-core / transformers，"
            "请安装项目 requirements.txt 后重试"
        ) from exc


_CONVERTER = None
_HF_TOKENIZER = None


class _MinLevelFilter(logging.Filter):
    """只放行 >= 指定级别的日志。"""

    def __init__(self, level: int) -> None:
        super().__init__()
        self._level = level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= self._level


def _quiet_ocr_logs() -> None:
    """压掉第三方 OCR 对“图上无字”区域的刷屏 warning。

    - docling 对每个 OCR 无结果的区域打一条 "RapidOCR returned empty result!"
    - RapidOCR 自己还会再打 "[RapidOCR] ... The text detection result is empty"

    图表 / 线条图区域没有文字是正常现象，不是失败；某文件的真实内容丢失应体现为
    该文件 chunk 数异常偏少。这里只把噪音压到 ERROR，真正的 OCR 错误仍会打印。

    注意：RapidOCR 的 Logger 在导入时会把自身 level 重置回 INFO，因此除 setLevel
    外再挂一个 filter —— filter 不受 setLevel 影响，与导入顺序无关地持续生效。
    """
    logging.getLogger("docling.models.stages.ocr.rapid_ocr_model").setLevel(
        logging.ERROR
    )

    rapidocr_logger = logging.getLogger("RapidOCR")
    rapidocr_logger.setLevel(logging.ERROR)
    if not any(isinstance(f, _MinLevelFilter) for f in rapidocr_logger.filters):
        rapidocr_logger.addFilter(_MinLevelFilter(logging.ERROR))


def _get_converter(DocumentConverter):
    """进程内复用 DocumentConverter。

    Docling 构造 converter 时会重建 PDF 管线并加载布局 / OCR 模型；每个文件新建
    一次会让 N 个文件重复加载 N 次模型（日志里的 "Loading weights" 进度条与
    RapidOCR 初始化块就是这么来的）。
    """
    global _CONVERTER
    if _CONVERTER is None:
        _CONVERTER = DocumentConverter()
    return _CONVERTER


def _get_hf_tokenizer(AutoTokenizer):
    """进程内复用 HuggingFace tokenizer。

    R3：local_files_only=True —— 只读本地缓存，离线环境不再触发 HF Hub 联网
    （否则每次重建 venv 都会 httpx.ConnectTimeout / WinError 10060）。
    """
    global _HF_TOKENIZER
    if _HF_TOKENIZER is None:
        _HF_TOKENIZER = AutoTokenizer.from_pretrained(
            settings.model.embedding.name,
            local_files_only=True,
        )
    return _HF_TOKENIZER


_quiet_ocr_logs()


def _normalize_headings(meta: Any) -> list[str]:
    """兼容不同 Docling 版本的 headings 表示。"""
    if meta is None:
        return []
    values = getattr(meta, "headings", None) or []
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text:
            result.append(text)
    return result


def _extract_page_range(meta: Any) -> tuple[int | None, int | None]:
    """尽量从 HybridChunker metadata 的 doc_items/prov 中恢复 PDF 页码。

    Docling 小版本间 metadata 结构可能变化，因此这里全部采用安全探测；
    取不到页码时返回 (None, None)，不影响正文定位与入库。
    """
    if meta is None:
        return None, None

    pages: list[int] = []
    doc_items = getattr(meta, "doc_items", None) or []
    for doc_item in doc_items:
        prov_items = getattr(doc_item, "prov", None) or []
        for prov in prov_items:
            page_no = getattr(prov, "page_no", None)
            if isinstance(page_no, int) and page_no > 0:
                pages.append(page_no)

    if not pages:
        return None, None
    return min(pages), max(pages)


def _prepare_docling_chunks(
    items: Iterable[Any],
    *,
    title: str,
) -> list[dict[str, Any]]:
    """把 Docling chunk 转为稳定的中间结构，过滤空文本/垃圾块/重复块。"""
    prepared: list[dict[str, Any]] = []
    seen_fingerprints: set[str] = set()

    for item in items:
        raw_text = getattr(item, "text", "") or ""
        text = clean_chunk_text(raw_text)
        if not text or is_junk_chunk(text):
            continue

        # 同一文档内完全重复的分块只保留第一次出现，避免页脚/表格冗余成片重复。
        fp = content_fingerprint(text)
        if fp in seen_fingerprints:
            continue
        seen_fingerprints.add(fp)

        meta = getattr(item, "meta", None)
        headings = _normalize_headings(meta)
        section_path = " / ".join(headings) if headings else title
        page, page_end = _extract_page_range(meta)

        prepared.append(
            {
                "text": text,
                "section_path": section_path,
                "page": page,
                "page_end": page_end,
            }
        )

    return prepared


def _page_aware_merge(
    prepared: list[dict[str, Any]],
    *,
    max_chars: int,
) -> list[dict[str, Any]]:
    """把同一页内相邻的过小 chunk 合并，且绝不跨页。

    HybridChunker(merge_peers=False) 会输出大量结构叶子小块（甚至单字符的
    页眉/作者行）。这里以「页」为硬边界，把同一页内相邻的小块拼回预算内的
    大块：既保留页码级引用（跨页绝不合并），又消除碎片化。
    """
    if not prepared:
        return []

    merged: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None

    for item in prepared:
        page = item.get("page")

        if (
            cur is not None
            and page is not None
            and cur.get("page") == page
            and cur.get("section_path") == item.get("section_path")
            and len(cur["text"]) + 2 + len(item["text"]) <= max_chars
        ):
            cur["text"] += "\n" + item["text"]
            cur["page_end"] = max(
                cur.get("page_end") or page,
                item.get("page_end") or page,
            )
        else:
            if cur is not None:
                merged.append(cur)
            cur = dict(item)
            cur["page"] = page
            cur["page_end"] = item.get("page_end") or page

    if cur is not None:
        merged.append(cur)

    return merged


def _docling_context_header(
    title: str,
    section_path: str,
    domain: str,
    subdomain: str,
) -> str:
    """Docling 路径的确定性上下文头（CCH），与 xlsx 行式路径的 _row_context_header 对齐。

    裸 chunk 文本嵌入时缺少「来自哪份文档/哪一章/哪个领域」的语境，这里用纯规则
    拼接一个简短头，随正文一起入库，让向量检索能消歧跨文档同名字段。
    """
    display_title = title.replace("_", " ").strip()
    parts = [f"文档《{display_title}》"]
    if section_path and section_path != title:
        parts.append(f"章节 {section_path}")
    parts.append(f"领域 {domain}/{subdomain}")
    return "；".join(parts) + "。"


def _chunks_and_units_from_prepared(
    prepared: list[dict[str, Any]],
    *,
    path: Path,
    domain: str,
    subdomain: str,
    document_type: str,
    source_file_hash: str,
    relative_path: str,
) -> tuple[list[Chunk], dict[str, str]]:
    """构造项目 Chunk，同时生成 SourceResolver 可精确恢复正文的 units。

    这里采用“一条 HybridChunker 输出 = 一个 source_unit”的方式。
    这样不会再依赖旧 parser 的字符坐标，同时保留 source_store 的
    index-only 安全架构：JSONL/Chroma 不保存正文，正文只存在 source_store。
    """
    document_id = stable_document_id(relative_path)
    title = path.stem
    source_format = path.suffix.lower().lstrip(".")

    section_counts = Counter(item["section_path"] for item in prepared)
    section_seen: Counter[str] = Counter()

    chunks: list[Chunk] = []
    units: dict[str, str] = {}

    for chunk_index, item in enumerate(prepared):
        text = item["text"]
        section_path = item["section_path"]

        # R4：Docling 以 page_no 表达幻灯片号；ppt 无物理页，page 应保持 None。
        page = item.get("page")
        page_end = item.get("page_end")
        slide_number = None
        if source_format == "pptx":
            slide_number = page
            page = None
            page_end = None

        # R1：确定性上下文头（CCH）烘焙进正文，与 xlsx 行式路径对齐。
        text = _docling_context_header(title, section_path, domain, subdomain) + "\n" + text

        # source_store 的 source_unit 是字符串键；每个 Docling chunk 独占一个 unit。
        unit_id = str(chunk_index + 1)
        units[unit_id] = text

        section_chunk_index = section_seen[section_path]
        section_seen[section_path] += 1
        section_chunk_count = section_counts[section_path]

        # 当前 unit 就是完整 chunk 文本，因此字符定位是 0..len(text)。
        char_start = 0
        char_end = len(text)
        line_start = 1
        line_end = text.count("\n") + 1
        last_line = text.rsplit("\n", 1)[-1]

        chunk_id = stable_chunk_id(document_id, chunk_index, text)

        chunks.append(
            Chunk(
                document_id=document_id,
                chunk_id=chunk_id,
                content=text,
                source_file_hash=source_file_hash,
                keywords=extract_keywords(text),
                domain=domain,
                subdomain=subdomain,
                document_type=document_type,
                source_file=path.name,
                relative_path=relative_path,
                source_format=source_format,
                title=title,
                section_path=section_path,
                page=page,
                page_end=page_end,
                section_id=f"SEC_{sha1_short(document_id + '|' + section_path)}",
                section_chunk_index=section_chunk_index,
                section_chunk_count=section_chunk_count,
                slide_number=slide_number,
                sheet_name=None,
                row_start=None,
                row_end=None,
                tags=["docling", "hybrid_chunker"],
                chunk_index=chunk_index,
                char_count=len(text),
                content_hash=digest(text),
                content_type="text",
                # source_store / SourceResolver 必需的精确定位字段
                source_unit=unit_id,
                char_start=char_start,
                char_end=char_end,
                line_start=line_start,
                line_end=line_end,
                column_start=1,
                column_end=len(last_line),
                line_basis="docling_chunk_v1",
            )
        )

    return chunks, units


def build_docling_chunks(
    path: Path,
    kb_root: Path,
    section_max_chars: int,
    domain: str,
    subdomain: str,
    document_type: str,
    source_file_hash: str,
    relative_path: str,
):
    """解析单文件并返回兼容原项目 Schema 的 Chunk 列表。

    kb_root / section_max_chars 保留在函数签名中，以兼容原程序入口；
    HybridChunker 的实际 token 上限统一由 settings.ingestion.docling.max_tokens 控制。
    """
    del kb_root  # 兼容旧调用签名；section_max_chars 现用于 _page_aware_merge 的同页重合并预算。

    (
        DocumentConverter,
        HybridChunker,
        HuggingFaceTokenizer,
        AutoTokenizer,
    ) = _docling_import()

    converter = _get_converter(DocumentConverter)
    result = converter.convert(str(path))
    document = result.document

    # 切片 tokenizer 与实际 Embedding 模型保持一致，避免 token 口径漂移。
    # （local_files_only 与进程内缓存见 _get_hf_tokenizer）
    hf_tokenizer = _get_hf_tokenizer(AutoTokenizer)
    docling_tokenizer = HuggingFaceTokenizer(
        tokenizer=hf_tokenizer,
        max_tokens=settings.ingestion.docling.max_tokens,
    )
    # R2：pdf/pptx 有页码/幻灯片定位，peer 合并会跨页合成 1 个 chunk（页码定位失效）。
    # 对这两类格式强制关闭 peer 合并，改由 _page_aware_merge 以「页」为硬边界重合并。
    merge_peers = settings.ingestion.docling.merge_peers
    if path.suffix.lower() in {".pdf", ".pptx"}:
        merge_peers = False

    chunker = HybridChunker(
        tokenizer=docling_tokenizer,
        merge_peers=merge_peers,
    )

    raw_chunks = list(chunker.chunk(document))
    prepared = _prepare_docling_chunks(raw_chunks, title=path.stem)
    prepared = _page_aware_merge(prepared, max_chars=section_max_chars)

    chunks, units = _chunks_and_units_from_prepared(
        prepared,
        path=path,
        domain=domain,
        subdomain=subdomain,
        document_type=document_type,
        source_file_hash=source_file_hash,
        relative_path=relative_path,
    )

    # 关键修复：这里必须保存 dict[str, str] 的 units，不能传空 list。
    # save_snapshot 会写 source_ref 回每个 chunk，随后 clean_chunk.py 才写 JSONL。
    save_snapshot(chunks, units)
    return chunks
