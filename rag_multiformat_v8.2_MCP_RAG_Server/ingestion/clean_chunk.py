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
from ingestion.source_store import save_snapshot, digest, file_digest, extract_keywords
from ingestion.docling_chunker import build_docling_chunks
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


def run(
    kb_root: Path,
    output_path: Path,
    error_path: Path,
    section_max_chars: int,
):
    if not kb_root.exists():
        raise FileNotFoundError(f"知识库目录不存在：{kb_root}")

    files = scan_supported_files(kb_root)

    validate_supported_file_domains(
        files=files,
        kb_root=kb_root,
    )

    print(f"支持格式：{', '.join(sorted(supported_extensions()))}")
    print(f"发现 {len(files)} 个可处理文件。")
    print(f"切片策略：Docling HybridChunker 结构化切片，"f"max_tokens={settings.ingestion.docling.max_tokens}。")



    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    ok_files = 0
    errors = []
    seen_fps: set = set()

    temp_output = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp_output.open("w", encoding="utf-8") as fout:
        for idx, path in enumerate(files, start=1):
            try:
                chunks = build_chunks_for_file(
                    path=path,
                    kb_root=kb_root,
                    section_max_chars=section_max_chars,
                )

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

                total_chunks += written
                ok_files += 1

                dup_note = f"（去重 {len(chunks) - written}）" if written < len(chunks) else ""
                print(
                    f"[{idx}/{len(files)}] OK  "
                    f"{path.relative_to(kb_root)} "
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
                    f"{path.relative_to(kb_root)} "
                    f"-> {type(e).__name__}（详情请在本地调试，日志不记录潜在正文）"
                )

    with error_path.open("w", encoding="utf-8") as ferr:
        for item in errors:
            ferr.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )

    if errors:
        temp_output.unlink(missing_ok=True)
        raise RuntimeError(f"{len(errors)} 个文件解析失败，未发布新索引，未执行向量入库。错误类型见 {error_path}")
    temp_output.replace(output_path)

    print("\n处理完成")
    print(f"成功文件：{ok_files}")
    print(f"失败文件：{len(errors)}")
    print(f"总 chunks：{total_chunks}")
    print(f"Chunk JSONL：{output_path}")
    print(f"错误日志：{error_path}")


def main():
    parser = argparse.ArgumentParser(
        description="多格式 / 多媒体知识库按章节清洗与切片"
    )

    parser.add_argument("--input", type=Path, default=settings.paths.kb_root)
    parser.add_argument("--output", type=Path, default=settings.paths.chunk_jsonl)
    parser.add_argument("--errors", type=Path, default=settings.paths.error_jsonl)

    args = parser.parse_args()

    run(
        kb_root=args.input,
        output_path=args.output,
        error_path=args.errors,
        section_max_chars=settings.ingestion.chunking.section_max_chars,
    )


if __name__ == "__main__":
    main()
