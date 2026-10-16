# -*- coding: utf-8 -*-
"""
阶段 1：
扫描整个知识库
-> 自动选择 Parser
-> 统一 CanonicalBlock
-> 保存解析文本原文快照
-> 按章节及页/行等位置边界切片
-> 无正文索引 JSONL

原则：
- 不跨章节合并；
- 同一章节内尽量合并为一个 Chunk；
- 章节过长时才做安全拆分；
- 不再使用字符 overlap。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import List, Tuple
from collections import Counter, defaultdict

from utils.config_loader import settings
from utils.schema import CanonicalBlock, Chunk
from parsers import get_parser, supported_extensions
from ingestion.source_store import (
    save_snapshot,
    digest,
    file_digest,
    extract_keywords,
)
from ingestion.docling_chunker import build_docling_chunks
from ingestion import index_state
from ingestion.chunk_clean import (
    clean_chunk_text,
    is_junk_chunk,
    content_fingerprint as chunk_fingerprint,
    normalize_text,
    strip_html_comments,
)


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；.!?;])\s*|\n+")


def sha1_short(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n].upper()


def stable_document_id(relative_path: str) -> str:
    return f"DOC_{sha1_short(relative_path)}"


def stable_chunk_id(document_id: str, chunk_index: int, content: str) -> str:
    return f"{document_id}_{chunk_index:04d}_{sha1_short(content, 8)}"


def stable_section_id(document_id: str, section_path: str) -> str:
    """同一文档同一章节共享稳定 section_id。"""
    return f"SEC_{sha1_short(document_id + '|' + section_path, 12)}"


def clean_text(text: str) -> str:
    """\u53bb HTML \u6ce8\u91ca + \u89c4\u8303\u5316\uff0c\u4e0e chunk_clean.normalize_text \u7edf\u4e00\u53e3\u5f84\u3002

    \u89c4\u8303\u5316\u903b\u8f91\u7edf\u4e00\u59d4\u6258\u7ed9 ingestion.chunk_clean.normalize_text\uff08BOM/NFC/CRLF/
    \u5236\u8868\u7b26/\u884c\u5c3e\u7a7a\u767d/\u8fde\u7eed\u7a7a\u884c\uff09\uff0c\u4ec5\u5728\u6b64\u4fdd\u7559\u539f clean_text \u7684 HTML \u6ce8\u91ca\u5265\u79bb\u884c\u4e3a\uff0c
    \u907f\u514d\u4e24\u5957\u5b9e\u73b0\u6f02\u79fb\u3002
    """
    return normalize_text(strip_html_comments(text))




def content_fingerprint(text: str) -> str:
    """用于块级去重的稳定指纹文本，不改变最终展示内容。"""
    text = clean_text(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def dedupe_section_text(text: str) -> str:
    """
    章节合并后的防御性去重。

    只删除同一章节内“完全重复”的段落/行，不做模糊改写，
    避免 Parser 异常、格式边界或源文件重复导致 Chunk 出现成片重复内容。
    """
    text = clean_text(text)
    if not text:
        return ""

    # 先按自然段去重，保留第一次出现的顺序。
    paragraphs = re.split(r"\n\s*\n", text)
    seen_paragraphs = set()
    kept_paragraphs = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        fp = content_fingerprint(paragraph)
        if fp in seen_paragraphs:
            continue

        seen_paragraphs.add(fp)
        kept_paragraphs.append(paragraph)

    text = "\n\n".join(kept_paragraphs)

    # 再删除完全重复的普通 prose/list 行。
    # fenced code、LaTeX display math、Markdown table 内部不做行级去重，
    # 避免破坏代码、公式和表格结构。
    seen_lines = set()
    kept_lines = []
    in_fence = False
    fence_token = None
    in_display_math = False

    for line in text.splitlines():
        stripped = line.strip()
        fp = re.sub(r"\s+", " ", stripped).lower()

        if stripped.startswith(("```", "~~~")):
            token = stripped[:3]
            if not in_fence:
                in_fence = True
                fence_token = token
            elif token == fence_token:
                in_fence = False
                fence_token = None
            kept_lines.append(line)
            continue

        if stripped == r"\[":
            in_display_math = True
            kept_lines.append(line)
            continue

        if stripped == r"\]":
            in_display_math = False
            kept_lines.append(line)
            continue

        protected = (
            in_fence
            or in_display_math
            or not stripped
            or stripped.startswith("|")
            or stripped in {"---", "***", "___"}
        )

        if not protected and len(fp) >= 4:
            if fp in seen_lines:
                continue
            seen_lines.add(fp)

        kept_lines.append(line)

    return clean_text("\n".join(kept_lines))


def infer_domain_subdomain(path: Path, kb_root: Path) -> Tuple[str, str]:
    rel = path.relative_to(kb_root)
    dirs = rel.parts[:-1]

    if not dirs:
        raise ValueError(
            "知识文件不能直接放在知识库根目录。"
            f"请放入一级领域目录：{', '.join(settings.domains.allowed)}"
        )

    domain = dirs[0].lower()

    if domain not in settings.domains.allowed:
        raise ValueError(
            f"不允许的一级 domain：{domain}。"
            f"仅允许：{', '.join(settings.domains.allowed)}。"
            "Datasheet、论文、设计案例、实验数据等应按实际知识归属"
            "放入 topology / device / pcb / control / schematic 之一。"
        )

    subdomain = dirs[1].lower() if len(dirs) >= 2 else "general"
    return domain, subdomain


def clone_block(block: CanonicalBlock, content: str) -> CanonicalBlock:
    return CanonicalBlock(
        content_type=block.content_type,
        content=content,
        section_path=block.section_path,
        page=block.page,
        page_end=block.page_end,
        sheet_name=block.sheet_name,
        row_start=block.row_start,
        row_end=block.row_end,
        slide_number=block.slide_number,
        image_index=block.image_index,
        media_start_sec=block.media_start_sec,
        media_end_sec=block.media_end_sec,
    )


def split_plain_text(text: str, max_chars: int) -> List[str]:
    """仅用于超长章节内部的安全拆分，不跨章节。"""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]

    if not sentences:
        return [text]

    pieces: List[str] = []
    current = ""

    for sentence in sentences:
        candidate = sentence if not current else current + "\n" + sentence

        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            pieces.append(current)
            current = ""

        if len(sentence) <= max_chars:
            current = sentence
            continue

        # 单个句子本身极长，只能硬拆。
        start = 0
        while start < len(sentence):
            pieces.append(sentence[start: start + max_chars])
            start += max_chars

    if current:
        pieces.append(current)

    return [p for p in pieces if p.strip()]


def split_structured_text(text: str, max_chars: int) -> List[str]:
    """表格/代码超长时按行安全拆分。"""
    lines = text.splitlines()
    result: List[str] = []
    buf: List[str] = []

    for line in lines:
        candidate = "\n".join(buf + [line])

        if buf and len(candidate) > max_chars:
            result.append("\n".join(buf).strip())
            buf = []

        if len(line) > max_chars:
            if buf:
                result.append("\n".join(buf).strip())
                buf = []

            start = 0
            while start < len(line):
                result.append(line[start: start + max_chars])
                start += max_chars
        else:
            buf.append(line)

    if buf:
        result.append("\n".join(buf).strip())

    return [x for x in result if x]


def expand_oversized_block(
    block: CanonicalBlock,
    max_chars: int,
) -> List[CanonicalBlock]:
    content = clean_text(block.content)
    if not content:
        return []

    base = clone_block(block, content)
    if len(content) <= max_chars:
        return [base]

    if block.content_type in {"table", "table_row", "code"}:
        pieces = split_structured_text(content, max_chars)
    else:
        pieces = split_plain_text(content, max_chars)

    return [clone_block(block, piece) for piece in pieces]


def section_key(block: CanonicalBlock, fallback_title: str) -> str:
    """章节是切片的一级边界；无章节结构时使用稳定的格式级回退。"""
    section = (block.section_path or "").strip()
    if section:
        return section

    if block.slide_number is not None:
        return f"Slide {block.slide_number}"
    if block.sheet_name:
        return block.sheet_name
    if block.page is not None:
        return f"Page {block.page}"
    return fallback_title


def merge_section_blocks(
    blocks: List[CanonicalBlock],
    fallback_title: str,
    max_chars: int,
) -> List[CanonicalBlock]:
    """
    章节优先切片：
    - section_path 不同 -> 必须切开；
    - section_path 相同 -> 尽量合并；
    - 同一章节超过 settings.ingestion.chunking.section_max_chars -> 章节内部安全拆分；
    - 不使用 overlap。
    """
    normalized: List[CanonicalBlock] = []
    seen_block_fingerprints = set()

    for block in blocks:
        section = section_key(block, fallback_title)
        normalized_block = clone_block(block, clean_text(block.content))
        normalized_block.section_path = section

        if not normalized_block.content:
            continue

        # 同一章节内完全重复的 Parser Block 只保留第一次出现。
        block_fp = (
            normalized_block.section_path,
            normalized_block.content_type,
            content_fingerprint(normalized_block.content),
        )
        if block_fp in seen_block_fingerprints:
            continue
        seen_block_fingerprints.add(block_fp)

        normalized.extend(
            expand_oversized_block(
                normalized_block,
                max_chars=max_chars,
            )
        )

    if not normalized:
        return []

    result: List[CanonicalBlock] = []
    current: List[CanonicalBlock] = []
    current_chars = 0
    current_section = ""

    def flush() -> None:
        nonlocal current, current_chars, current_section

        if not current:
            return

        contents = [b.content.strip() for b in current if b.content.strip()]
        text = dedupe_section_text("\n\n".join(contents))

        if text:
            content_types = {b.content_type for b in current}

            page_values = []
            for b in current:
                if b.page is not None:
                    page_values.append(b.page)
                if b.page_end is not None:
                    page_values.append(b.page_end)

            row_starts = [b.row_start for b in current if b.row_start is not None]
            row_ends = [b.row_end for b in current if b.row_end is not None]
            media_starts = [
                b.media_start_sec for b in current if b.media_start_sec is not None
            ]
            media_ends = [
                b.media_end_sec for b in current if b.media_end_sec is not None
            ]

            slide_values = [
                b.slide_number for b in current if b.slide_number is not None
            ]
            image_values = [
                b.image_index for b in current if b.image_index is not None
            ]
            sheet_values = [
                b.sheet_name for b in current if b.sheet_name
            ]

            result.append(
                CanonicalBlock(
                    content_type=(
                        next(iter(content_types))
                        if len(content_types) == 1
                        else "mixed"
                    ),
                    content=text,
                    section_path=current_section,
                    page=min(page_values) if page_values else None,
                    page_end=max(page_values) if page_values else None,
                    sheet_name=(
                        sheet_values[0]
                        if sheet_values and len(set(sheet_values)) == 1
                        else None
                    ),
                    row_start=min(row_starts) if row_starts else None,
                    row_end=max(row_ends) if row_ends else None,
                    slide_number=(
                        slide_values[0]
                        if slide_values and len(set(slide_values)) == 1
                        else None
                    ),
                    image_index=(
                        image_values[0]
                        if image_values and len(set(image_values)) == 1
                        else None
                    ),
                    media_start_sec=min(media_starts) if media_starts else None,
                    media_end_sec=max(media_ends) if media_ends else None,
                )
            )

        current = []
        current_chars = 0
        current_section = ""

    for block in normalized:
        if not current:
            current = [block]
            current_chars = len(block.content)
            current_section = block.section_path
            continue

        # 不跨章节。
        if block.section_path != current_section:
            flush()
            current = [block]
            current_chars = len(block.content)
            current_section = block.section_path
            continue

        extra_chars = 2 + len(block.content)

        # 同章节内优先合并；超过安全长度时切分为“章节的下一部分”。
        if current_chars + extra_chars > max_chars:
            flush()
            current = [block]
            current_chars = len(block.content)
            current_section = block.section_path
            continue

        current.append(block)
        current_chars += extra_chars

    flush()
    return result


def _row_context_header(
    title: str,
    sheet_name: str | None,
    domain: str,
    subdomain: str,
) -> str:
    """行式 chunk 的确定性上下文头（CCH）。

    裸行文本（如 ``Package: SOIC-8；VDS: 80``）在嵌入时缺少“来自哪份文档、
    哪张工作表、哪个领域”的语境。这里用纯规则拼接一个简短头，随正文一起入库，
    让向量检索能消歧同名字段、跨文档对齐。

    注意：本项目“索引不存正文”的架构没有单独的 embed-text 字段，因此上下文头
    直接烘焙进 chunk.content，source_store 的 unit 同样包含该头，roundtrip 保持一致。
    """
    display_title = title.replace("_", " ").strip()
    parts = [f"文档《{display_title}》"]
    if sheet_name:
        parts.append(f"工作表「{sheet_name}」")
    parts.append(f"领域 {domain}/{subdomain}")
    return "；".join(parts) + "。"


def _blocks_to_chunks(
    merged: List[CanonicalBlock],
    *,
    path: Path,
    kb_root: Path,
    source_file_hash: str,
) -> List[Chunk]:
    """把合并后的 CanonicalBlock 转为项目 Chunk + source_store units。"""
    relative_path = path.relative_to(kb_root).as_posix()
    document_id = stable_document_id(relative_path)
    title = path.stem
    source_format = path.suffix.lower().lstrip(".")
    document_type = path.suffix.lower().lstrip(".")
    domain, subdomain = infer_domain_subdomain(path, kb_root)

    section_counts = Counter(block.section_path for block in merged)
    section_seen: Counter[str] = Counter()

    chunks: List[Chunk] = []
    units: dict[str, str] = {}

    for chunk_index, block in enumerate(merged):
        text = block.content
        header = _row_context_header(title, block.sheet_name, domain, subdomain)
        if header:
            text = header + "\n" + text
        unit_id = str(chunk_index + 1)
        units[unit_id] = text

        section_path = block.section_path or title
        section_chunk_index = section_seen[section_path]
        section_seen[section_path] += 1
        section_chunk_count = section_counts[section_path]

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
                page=block.page,
                page_end=block.page_end,
                sheet_name=block.sheet_name,
                row_start=block.row_start,
                row_end=block.row_end,
                image_index=block.image_index,
                media_start_sec=block.media_start_sec,
                media_end_sec=block.media_end_sec,
                tags=[block.content_type or "table_row"],
                chunk_index=chunk_index,
                char_count=len(text),
                content_hash=digest(text),
                content_type=block.content_type or "table_row",
                section_id=f"SEC_{sha1_short(document_id + '|' + section_path)}",
                section_chunk_index=section_chunk_index,
                section_chunk_count=section_chunk_count,
                source_unit=unit_id,
                char_start=0,
                char_end=len(text),
                line_start=1,
                line_end=text.count("\n") + 1,
                column_start=1,
                column_end=len(last_line),
                line_basis="row_chunk_v1",
            )
        )

    save_snapshot(chunks, units)
    return chunks


def build_row_chunks_for_file(
    path: Path,
    kb_root: Path,
    section_max_chars: int,
) -> List[Chunk]:
    """专用解析器路径：行式格式（xlsx/xlsm/csv）与多媒体（音频/视频/图片）。

    行式格式每行一条记录，按 sheet 合并为不超过 section_max_chars 的 chunk，
    恢复 sheet_name / row_start / row_end 定位，绕过 Docling 对表格的展平与成片重复；
    多媒体格式由 audio/video/image parser 做 ASR / OCR，恢复 media_start_sec /
    image_index 定位（Docling 无 ASR，多媒体不能走 Docling）。
    """
    parser = get_parser(path)
    if parser is None:
        raise RuntimeError(f"行式格式缺少解析器：{path.suffix}")

    source_file_hash = file_digest(path)
    blocks = parser.parse(path)
    if not blocks:
        return []

    # 清洗 + 过滤垃圾行 + 行级去重。
    cleaned: List[CanonicalBlock] = []
    seen_rows: set = set()
    for block in blocks:
        text = clean_chunk_text(block.content)
        if not text or is_junk_chunk(text):
            continue
        fp = chunk_fingerprint(text)
        if fp in seen_rows:
            continue
        seen_rows.add(fp)
        cleaned.append(clone_block(block, text))

    if not cleaned:
        return []

    merged = merge_section_blocks(cleaned, path.stem, section_max_chars)
    chunks = _blocks_to_chunks(
        merged,
        path=path,
        kb_root=kb_root,
        source_file_hash=source_file_hash,
    )

    if file_digest(path) != source_file_hash:
        raise RuntimeError("解析期间源文件发生变化，请重新入库")

    return chunks


def build_chunks_for_file(
    path: Path,
    kb_root: Path,
    section_max_chars: int,
) -> List[Chunk]:
    """
    Docling HybridChunker版本:
    - 多格式解析统一交给Docling
    - 使用HybridChunker进行结构感知切片
    - 保留原Chunk Schema，兼容Embedding/Search/Eval
    - 行式格式（xlsx/xlsm/csv）改走行感知切片，恢复 row 定位
    """

    suffix = path.suffix.lower()
    parser = get_parser(path)
    # 行式格式（xlsx/xlsm/csv）与多媒体格式（音频/视频/图片）走专用解析器。
    # Docling 无法做 ASR / 媒体 OCR，多媒体必须由 audio/video/image parser 转录，
    # 否则转录内容会被静默丢失，或 DocumentConverter().convert() 抛异常导致整个入库中止。
    if suffix in {".xlsx", ".xlsm", ".csv"} or (
        parser is not None
        and getattr(parser, "document_type", "") in {"audio", "video", "image"}
    ):
        return build_row_chunks_for_file(path, kb_root, section_max_chars)

    source_file_hash = file_digest(path)

    relative_path = path.relative_to(kb_root).as_posix()
    document_type = path.suffix.lower().lstrip(".")

    # 保持原项目domain规则
    domain, subdomain = infer_domain_subdomain(path, kb_root)

    chunks = build_docling_chunks(
        path=path,
        kb_root=kb_root,
        section_max_chars=section_max_chars,
        domain=domain,
        subdomain=subdomain,
        document_type=document_type,
        source_file_hash=source_file_hash,
        relative_path=relative_path,
    )

    if file_digest(path) != source_file_hash:
        raise RuntimeError("解析期间源文件发生变化，请重新入库")

    return chunks


def scan_supported_files(kb_root: Path) -> List[Path]:
    exts = supported_extensions()

    return sorted(
        p
        for p in kb_root.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    )


def validate_supported_file_domains(
    files: List[Path],
    kb_root: Path,
) -> None:
    invalid_items = []

    for path in files:
        rel = path.relative_to(kb_root)
        dirs = rel.parts[:-1]

        if not dirs:
            invalid_items.append(
                f"{rel.as_posix()} -> 缺少一级 domain"
            )
            continue

        domain = dirs[0].lower()

        if domain not in settings.domains.allowed:
            invalid_items.append(
                f"{rel.as_posix()} -> domain={domain}"
            )

    if invalid_items:
        preview = "\n".join(
            f"- {item}"
            for item in invalid_items[:20]
        )

        more = (
            f"\n... 另有 {len(invalid_items) - 20} 个非法文件"
            if len(invalid_items) > 20
            else ""
        )

        raise ValueError(
            "知识库存在不允许的一级 domain。\n"
            f"仅允许：{', '.join(settings.domains.allowed)}\n"
            "Datasheet、论文、设计案例、实验数据等资料类型，"
            "应根据知识归属放入上述五个领域之一。\n"
            + preview
            + more
        )


class ChunkingContractError(RuntimeError):
    """切片结果违反了增量所依赖的不变量（见 index_state.normalized_stem）。"""


def _write_file_rows(fout, chunks, seen_fps: set) -> int:
    """写一个文件的 chunk 行，跨文件去重后返回实际写入条数。

    全量与增量共用这一份实现：去重口径只能有一处，否则两种模式的结果会漂移。
    """
    written = 0
    for chunk in chunks:
        fp = chunk_fingerprint(chunk.content)
        if fp in seen_fps:
            continue
        seen_fps.add(fp)
        fout.write(
            json.dumps(
                chunk.to_index_dict(),
                ensure_ascii=False,
            )
            + "\n"
        )
        written += 1
    return written


def _write_errors(error_path: Path, errors: List[dict]) -> None:
    with error_path.open("w", encoding="utf-8") as ferr:
        for item in errors:
            ferr.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )


def _file_stamp(path: Path):
    """发布前复核用的文件戳；文件不存在时返回 None。"""
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def _remember_pipeline_state(output_path: Path, *, verified: bool, **extra) -> None:
    """记录切片口径指纹（以及可选的本次摘要）。

    全量与增量都要写：不写的话下一次增量拿不到基线指纹，只能退化成「以现有索引
    为准 + 告警」，等于把「切片代码/配置是否变过」这个判断永久放弃。

    ``verified=False`` 表示这份指纹是「从一份来历不明的既有索引直接接管」得到的，
    并没有真正校验过该索引是否由当前口径产出。它只用于让告警持续可见——不能因为
    记了指纹就当已经确认过。
    """
    try:
        index_state.write_state(
            index_state.state_path(output_path),
            {
                "chunk_pipeline_version": index_state.CHUNK_PIPELINE_VERSION,
                "pipeline_fingerprint": index_state.pipeline_fingerprint(),
                # 存原始输入项而不只是哈希：指纹的覆盖范围本身会演进，
                # 只有能逐项比对，才能区分「口径定义变了」与「输入真的变了」。
                "fingerprint_parts": index_state.fingerprint_parts(),
                "fingerprint_verified": bool(verified),
                **extra,
            },
        )
    except OSError as exc:
        print(f"[WARN] 入库状态文件写入失败（不影响本次结果）：{exc}")


def _discard_temp_state(temp_output: Path) -> None:
    """发布后清掉 .tmp 的伴随状态。

    留着它会让下一次 --resume 看到一份陈旧的 complete 标记，从而误判手上这份 .tmp
    是完整的中断产物（或用相反的判断把完整数据当成半截丢掉）。
    """
    try:
        index_state.state_path(temp_output).unlink(missing_ok=True)
    except OSError:
        pass


def run(
    kb_root: Path,
    output_path: Path,
    error_path: Path,
    section_max_chars: int,
    resume: bool = False,
    incremental: bool = False,
    dry_run: bool = False,
    force_delete: bool = False,
) -> dict:
    if not kb_root.exists():
        raise FileNotFoundError(f"知识库目录不存在：{kb_root}")

    if resume and incremental:
        raise ValueError(
            "--resume 与 --incremental 不能同时使用：两者只是「基线取自哪里」不同"
            "（--resume 优先用中断的 .tmp，--incremental 用已发布的索引），选一个即可。"
        )

    files = scan_supported_files(kb_root)

    validate_supported_file_domains(
        files=files,
        kb_root=kb_root,
    )

    print(f"支持格式：{', '.join(sorted(supported_extensions()))}")
    print(f"发现 {len(files)} 个可处理文件。")

    # --resume 复用增量那一整套装配逻辑，只是把基线从「已发布的索引」换成「中断的 .tmp」。
    # 这样路径顺序、跨文件去重、孪生分组都只有一份实现，续跑结果必然等于全量重跑。
    if incremental or dry_run or resume:
        return _run_incremental(
            kb_root=kb_root,
            output_path=output_path,
            error_path=error_path,
            section_max_chars=section_max_chars,
            files=files,
            dry_run=dry_run,
            force_delete=force_delete,
            resume=resume,
        )

    print(f"切片策略：Docling HybridChunker 结构化切片，"f"max_tokens={settings.ingestion.docling.max_tokens}。")



    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)

    temp_output = output_path.with_suffix(output_path.suffix + ".tmp")

    # 把本次的切片口径指纹钉在 .tmp 旁边：中断后 --resume 才能确认「这份中断产物
    # 是不是当前配置产出的」，否则续跑可能把两种口径的切片混在一起。
    # complete 显式置 False：状态文件是合并写入的，不写就会残留上一次运行的 True。
    _remember_pipeline_state(temp_output, verified=True, complete=False)

    total_chunks = 0
    ok_files = 0
    errors = []
    seen_fps: set = set()

    with temp_output.open("w", encoding="utf-8") as fout:
        for idx, path in enumerate(files, start=1):
            relative = path.relative_to(kb_root).as_posix()
            try:
                chunks = build_chunks_for_file(
                    path=path,
                    kb_root=kb_root,
                    section_max_chars=section_max_chars,
                )

                written = _write_file_rows(fout, chunks, seen_fps)

                total_chunks += written
                ok_files += 1

                dup_note = f"（去重 {len(chunks) - written}）" if written < len(chunks) else ""
                print(
                    f"[{idx}/{len(files)}] OK  "
                    f"{relative} "
                    f"-> {written} section chunks{dup_note}"
                )

            except Exception as e:
                item = {
                    "file": str(path),
                    "error_type": type(e).__name__,
                }

                errors.append(item)

                print(
                    f"[{idx}/{len(files)}] ERR "
                    f"{relative} "
                    f"-> {type(e).__name__}（详情请在本地调试，日志不记录潜在正文）"
                )

    # with 块退出即关闭文件、落盘全部缓冲：此刻 .tmp 对每个处理过的文件都是完整的。
    # 打上这个标记，续跑才能区分「Python 异常退出（数据完整）」与「进程被原生杀掉
    # （最后一个文件可能只写了一半）」，从而决定要不要丢弃最后一组。
    _remember_pipeline_state(temp_output, verified=True, complete=True)

    _write_errors(error_path, errors)

    if errors:
        # 保留 .tmp 不删除：其中已完成文件的记录可被下次 --resume 复用，
        # 避免一次失败让整轮数小时的切片工作作废。索引仍然不发布。
        raise RuntimeError(
            f"{len(errors)} 个文件解析失败，未发布新索引，未执行向量入库。"
            f"错误类型见 {error_path}；已完成的切片已保留在 {temp_output.name}，"
            "修复后可加 --resume 继续。"
        )
    temp_output.replace(output_path)
    _discard_temp_state(temp_output)
    # 全量重切过，索引口径与当前代码/配置一致，指纹可以认为已确认。
    _remember_pipeline_state(output_path, verified=True)

    print("\n处理完成")
    print(f"成功文件：{ok_files}")
    print(f"失败文件：{len(errors)}")
    print(f"总 chunks：{total_chunks}")
    print(f"Chunk JSONL：{output_path}")
    print(f"错误日志：{error_path}")

    return {
        "mode": "full",
        "files": len(files),
        "chunks": total_chunks,
        "errors": len(errors),
    }


def _describe_plan(
    plan: index_state.Plan,
    files: List[Path],
    kb_root: Path,
    *,
    label: str = "incremental",
    limit: int = 20,
) -> None:
    def show(title: str, items: List[str]) -> None:
        if not items:
            return
        print(f"[{label}] {title}（{len(items)}）:")
        for item in items[:limit]:
            print(f"  - {item}")
        if len(items) > limit:
            print(f"  ... 另有 {len(items) - limit} 个")

    print(
        f"[{label}] 新增 {len(plan.added)}，重切 {len(plan.changed)}，"
        f"删除 {len(plan.deleted_paths)}，未变 {len(plan.unchanged)}"
    )
    show("将新增/重切", sorted({path.relative_to(kb_root).as_posix() for path in plan.reparse}))
    show("将删除", plan.deleted_paths)
    for warning in plan.warnings:
        print(f"[{label}][WARN] {warning}")


def _run_incremental(
    *,
    kb_root: Path,
    output_path: Path,
    error_path: Path,
    section_max_chars: int,
    files: List[Path],
    dry_run: bool,
    force_delete: bool,
    resume: bool = False,
) -> dict:
    """增量切片：只重新解析新增/变化的文件，复用未变文件的原始行。

    索引仍以「整份原子重写」的方式发布——未变行逐字节复制，变化行重新生成——
    因此发布出来的索引永远等于「当前知识库的完整切片」，下游（词法检索、评估脚本、
    审计）的「行集 == 语料全集」假设继续成立。

    ``resume=True`` 只是把基线从「已发布的索引」换成「中断的 .tmp」（没有则退回已发布
    索引），其余逻辑完全一致——包括按路径顺序装配、跨文件去重、孪生分组。于是「续跑」
    与「全量重跑」必然逐字节相同，而且中断期间增删改过的文件也会被正确处理（旧实现按
    路径跳过，改过的文件会被当成已完成，且靠后文件会抢走跨文件去重的胜负）。
    """
    label = "resume" if resume else "incremental"
    temp_output = output_path.with_suffix(output_path.suffix + ".tmp")

    if resume:
        if temp_output.exists() and temp_output.stat().st_size > 0:
            base = temp_output
            # 只有「进程被原生杀掉」才需要丢弃最后一组：那种情况下最后一行可能只写了一半。
            # 若是 Python 异常退出，with 块退出时已把缓冲全部落盘，数据是完整的。
            complete = index_state.read_state(
                index_state.state_path(temp_output)
            ).get("complete")
            drop_last = not complete
        elif output_path.exists() and output_path.stat().st_size > 0:
            base, drop_last = output_path, False
        else:
            base, drop_last = None, False
    else:
        usable = output_path.exists() and output_path.stat().st_size > 0
        base, drop_last = (output_path, False) if usable else (None, False)

    if base is None:
        if dry_run:
            print(f"[{label}] 没有可复用的基线，dry-run 无内容可预览。")
            return {"mode": label, "dry_run": True, "added": len(files),
                    "changed": 0, "deleted": 0, "unchanged": 0}
        print(f"[{label}] 没有可复用的基线，本次按全量处理。")
        return run(
            kb_root=kb_root,
            output_path=output_path,
            error_path=error_path,
            section_max_chars=section_max_chars,
        )

    print(f"[{label}] 基线：{base.name}")

    # 指纹从**基线自己的伴随状态**读：.tmp 的伴随状态由产生它的那次运行写在旁边，
    # 所以「中断后改了切片配置再续跑」会被拦下，而不是把两种口径混在一起。
    state = index_state.read_state(index_state.state_path(base))
    remembered = state.get("pipeline_fingerprint")
    verified = state.get("fingerprint_verified") is True
    current = index_state.pipeline_fingerprint()
    if remembered is None:
        # 基线来历不明（首次增量接管的旧索引，或旧版本产生的中断产物）：无法校验它是否由
        # 当前切片口径产出。不阻断（否则第一次增量就被迫全量重切），但也绝不把现在的指纹
        # 当成「已确认」，让告警一直挂到用户跑过一次全量为止。
        verified = False
        print(
            f"[{label}][WARN] 基线缺少流水线指纹（首次增量，或状态文件被删）。"
            f"本次以 {base.name} 为基线，但无法确认它是否由当前切片代码/配置产出；"
            "若其间改过切片配置或解析器版本，请先全量重跑。此告警会持续到跑过一次全量。"
        )
    elif remembered != current:
        changed = index_state.fingerprint_diff(
            state.get("fingerprint_parts") or {}, index_state.fingerprint_parts()
        )
        if not changed:
            # 两边同名输入项取值完全一致，只是指纹的覆盖范围变了（新增/移除了纳入的键）。
            # 这不是语料口径变化，不必逼用户重切几小时；但也不能当作已校验。
            verified = False
            print(
                f"[{label}][WARN] 指纹的覆盖范围已更新，但可比对的输入项全部一致。"
                "本次继续；跑过一次全量后此告警消失。"
            )
        else:
            raise RuntimeError(
                "切片口径已变化（" + "、".join(changed) + f"），继续用 {base.name} 会把新旧"
                "两种口径的 chunk 混在一起。\n请先全量重跑一次：python -m ingestion.ingest"
            )
    elif not verified:
        print(
            f"[{label}][WARN] 当前基线是由一份来历不明的产物直接接管的，从未校验过。"
            "跑过一次全量后此告警消失。"
        )

    snapshot = index_state.scan_index(base, drop_last_file=drop_last)
    if snapshot.tail_truncated and not drop_last:
        raise RuntimeError(
            f"{base.name} 的最后一行不完整（写入被中断），无法作为基线，请全量重跑。"
        )
    if snapshot.schema_mismatch:
        raise RuntimeError("索引中存在非当前 index_schema_version 的行，请全量重跑。")
    if snapshot.duplicates:
        raise RuntimeError(
            "索引中存在重复 chunk_id：" + ", ".join(snapshot.duplicates) + "，请全量重跑。"
        )

    # 闸门一：扫描为空但索引非空 —— 知识库目录多半不可达（网络盘掉线/权限/盘符未挂载）。
    # 不加这道闸，一次掉线就会把索引和向量库一起清空。
    if not files and snapshot.paths:
        raise RuntimeError(
            f"知识库扫描结果为 0 个文件，但索引里有 {len(snapshot.paths)} 个文件。\n"
            "这通常意味着知识库目录不可达。如果确实要清空索引，请手动删除：\n"
            f"  {output_path}"
        )

    plan = index_state.classify(kb_root, files, snapshot)

    # 闸门二：单次删除占比过高时需要显式确认。
    ratio = len(plan.deleted_paths) / max(len(snapshot.paths), 1)
    max_ratio = settings.ingestion.incremental.max_delete_ratio
    if plan.deleted_paths and ratio > max_ratio and not force_delete:
        raise RuntimeError(
            f"本次将删除 {len(plan.deleted_paths)} 个文件（占索引 {ratio:.0%}，"
            f"超过阈值 {max_ratio:.0%}），已中止以免误删。\n"
            "确认无误请加 --force-delete 重跑；不确定就先跑 --dry-run 看清单。"
        )

    _describe_plan(plan, files, kb_root, label=label)

    if dry_run:
        return {
            "mode": label,
            "dry_run": True,
            "added": len(plan.added),
            "changed": len(plan.changed),
            "deleted": len(plan.deleted_paths),
            "unchanged": len(plan.unchanged),
        }

    # 增量下「无变化」可以直接返回：已发布的索引内容本来就是对的。
    # 但续跑不行——基线是还没发布的 .tmp，必须走完装配并发布，否则这次续跑等于白跑。
    if plan.is_empty and not resume:
        # 不建 .tmp、不碰索引：一碰 mtime，长驻服务的词法索引就会白重建一次。
        error_path.parent.mkdir(parents=True, exist_ok=True)
        _write_errors(error_path, [])
        print(f"[{label}] 没有检测到任何变化，索引保持原样。")
        _remember_pipeline_state(
            output_path,
            verified=verified,
            last_run={"added": 0, "changed": 0, "deleted": 0, "empty_files": []},
        )
        return {"mode": label, "added": 0, "changed": 0, "deleted": 0,
                "unchanged": len(plan.unchanged), "chunks": 0, "errors": 0}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)

    if resume and temp_output.exists():
        print(f"[{label}] 覆盖 {temp_output.name}（其中未变的文件会被原样写入新索引）。")

    baseline = _file_stamp(output_path)
    unchanged_map = {
        path.relative_to(kb_root).as_posix(): lines for path, lines in plan.unchanged
    }
    reparse = {path.relative_to(kb_root).as_posix() for path in plan.reparse}

    seen_fps: set = set()
    total_chunks = 0
    ok_files = 0
    empty_files: List[str] = []
    errors = []

    with temp_output.open("w", encoding="utf-8") as fout:
        for idx, path in enumerate(files, start=1):
            relative = path.relative_to(kb_root).as_posix()

            if relative in unchanged_map:
                # 逐字节写回：不重新序列化，索引才可能在下游 diff 里保持稳定。
                for line in unchanged_map[relative]:
                    fout.write(line + "\n")
                total_chunks += len(unchanged_map[relative])
                ok_files += 1
                continue

            if relative not in reparse:
                continue

            try:
                chunks = build_chunks_for_file(
                    path=path,
                    kb_root=kb_root,
                    section_max_chars=section_max_chars,
                )

                # 孪生分组的充分性依赖这个不变量；不成立就必须全量，否则跨文件去重
                # 可能跨 stem 碰撞而无人察觉（详见 index_state.normalized_stem）。
                if chunks and not index_state.cch_invariant_holds(chunks):
                    raise ChunkingContractError(
                        f"{relative} 的切片结果没有以确定性上下文头（CCH）开头。"
                    )

                written = _write_file_rows(fout, chunks, seen_fps)
                total_chunks += written
                ok_files += 1
                if written == 0:
                    empty_files.append(relative)

                dup_note = f"（去重 {len(chunks) - written}）" if written < len(chunks) else ""
                print(f"[{idx}/{len(files)}] OK  {relative} -> {written} section chunks{dup_note}")

            except ChunkingContractError:
                # 不变量被破坏属于全局问题（切片器改动），不能降级成单文件错误。
                raise
            except Exception as e:
                errors.append({"file": str(path), "error_type": type(e).__name__})
                print(
                    f"[{idx}/{len(files)}] ERR {relative} -> {type(e).__name__}"
                    "（详情请在本地调试，日志不记录潜在正文）"
                )

    # 同全量：with 块退出即全部落盘，这份 .tmp 对处理过的文件是完整的。
    _remember_pipeline_state(temp_output, verified=verified, complete=True)

    _write_errors(error_path, errors)

    if errors:
        raise RuntimeError(
            f"{len(errors)} 个文件解析失败，未发布新索引，未执行向量入库。"
            f"错误类型见 {error_path}；已完成的切片已保留在 {temp_output.name}。"
            f"修复后重跑 --{label} 即可（未变的文件不会被重新解析）。"
        )

    # 发布前复核基线：期间若有别的进程改过索引，本次的未变行基线就已失效。
    if _file_stamp(output_path) != baseline:
        raise RuntimeError(
            "索引在本次运行期间被其他进程改动，已放弃发布以免覆盖对方的更新。\n"
            "请确认没有并发的 ingest 后重跑。"
        )

    temp_output.replace(output_path)
    _discard_temp_state(temp_output)

    # sidecar 在发布之后写，且只是 advisory：写坏了最坏让下次多跑一次全量判定。
    _remember_pipeline_state(
        output_path,
        verified=verified,
        last_run={
            "added": len(plan.added),
            "changed": len(plan.changed),
            "deleted": len(plan.deleted_paths),
            "empty_files": sorted(set(empty_files)),
        },
    )

    print(f"\n处理完成（{label}）")
    print(f"成功文件：{ok_files}（其中重切 {len(plan.reparse)}）")
    print(f"失败文件：{len(errors)}")
    print(f"总 chunks：{total_chunks}")
    print(f"Chunk JSONL：{output_path}")
    print(f"错误日志：{error_path}")

    return {
        "mode": label,
        "added": len(plan.added),
        "changed": len(plan.changed),
        "deleted": len(plan.deleted_paths),
        "unchanged": len(plan.unchanged),
        "chunks": total_chunks,
        "errors": len(errors),
    }


def main():
    parser = argparse.ArgumentParser(
        description="多格式 / 多媒体知识库按章节清洗与切片"
    )

    parser.add_argument("--input", type=Path, default=settings.paths.kb_root)
    parser.add_argument("--output", type=Path, default=settings.paths.chunk_jsonl)
    parser.add_argument("--errors", type=Path, default=settings.paths.error_jsonl)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从中断的 .tmp（没有则用已发布索引）继续；未变的文件直接复用，结果与全量重跑一致",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="只重新解析新增/内容变化的文件，并删除已不存在文件的 chunk",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只列出将会新增/重切/删除的文件，不写任何东西",
    )
    parser.add_argument(
        "--force-delete",
        action="store_true",
        help="允许单次删除超过阈值比例的文件（默认拒绝，防误删）",
    )

    args = parser.parse_args()

    run(
        kb_root=args.input,
        output_path=args.output,
        error_path=args.errors,
        section_max_chars=settings.ingestion.chunking.section_max_chars,
        resume=args.resume,
        incremental=args.incremental,
        dry_run=args.dry_run,
        force_delete=args.force_delete,
    )


if __name__ == "__main__":
    main()
