# -*- coding: utf-8 -*-
"""
阶段 1：
扫描整个知识库
-> 自动选择 Parser
-> 统一 CanonicalBlock
-> 清洗
-> 切片
-> 标准 Chunk JSONL

正常使用时由 ingest.py 调用。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import List, Tuple

from config import (
    KB_ROOT,
    CHUNK_JSONL,
    ERROR_JSONL,
    MAX_CHARS,
    OVERLAP_CHARS,
    HARD_BLOCK_MAX_CHARS,
)
from schema import CanonicalBlock, Chunk
from parsers import get_parser, supported_extensions


_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_TRAILING_SPACES_RE = re.compile(r"[ \t]+$", re.M)
_EXCESS_BLANK_RE = re.compile(r"\n{3,}")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；.!?;])\s*|\n+")


def sha1_short(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n].upper()


def stable_document_id(relative_path: str) -> str:
    return f"DOC_{sha1_short(relative_path)}"


def stable_chunk_id(document_id: str, chunk_index: int, content: str) -> str:
    return f"{document_id}_{chunk_index:04d}_{sha1_short(content, 8)}"


def clean_text(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HTML_COMMENT_RE.sub("", text)
    text = text.replace("\t", "    ")
    text = _TRAILING_SPACES_RE.sub("", text)
    text = _EXCESS_BLANK_RE.sub("\n\n", text)
    return text.strip()


def infer_domain_subdomain(path: Path, kb_root: Path) -> Tuple[str, str]:
    rel = path.relative_to(kb_root)
    dirs = rel.parts[:-1]

    domain = dirs[0].lower() if len(dirs) >= 1 else "unknown"
    subdomain = dirs[1].lower() if len(dirs) >= 2 else "general"

    return domain, subdomain


def split_plain_text(text: str, max_chars: int) -> List[str]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]

    if not sentences:
        return [text]

    units: List[str] = []
    current = ""

    for sentence in sentences:
        candidate = sentence if not current else current + "\n" + sentence

        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            units.append(current)
            current = ""

        if len(sentence) > max_chars:
            start = 0

            while start < len(sentence):
                units.append(sentence[start:start + max_chars])
                start += max_chars
        else:
            current = sentence

    if current:
        units.append(current)

    return units


def split_structured_block(text: str, max_chars: int) -> List[str]:
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
                result.append(line[start:start + max_chars])
                start += max_chars

        else:
            buf.append(line)

    if buf:
        result.append("\n".join(buf).strip())

    return [x for x in result if x]


def clone_block(block: CanonicalBlock, content: str) -> CanonicalBlock:
    return CanonicalBlock(
        content_type=block.content_type,
        content=content,
        section_path=block.section_path,

        page=block.page,

        sheet_name=block.sheet_name,
        row_start=block.row_start,
        row_end=block.row_end,

        slide_number=block.slide_number,
        image_index=block.image_index,

        media_start_sec=block.media_start_sec,
        media_end_sec=block.media_end_sec,
    )


def block_to_units(block: CanonicalBlock, max_chars: int) -> List[CanonicalBlock]:
    content = clean_text(block.content)

    if not content:
        return []

    base = clone_block(block, content)

    if len(content) <= HARD_BLOCK_MAX_CHARS:
        return [base]

    if block.content_type in {
        "table",
        "code",
        "table_row",
    }:
        pieces = split_structured_block(content, max_chars)
    else:
        pieces = split_plain_text(content, max_chars)

    return [clone_block(block, piece) for piece in pieces]


def same_context(a: CanonicalBlock, b: CanonicalBlock) -> bool:
    """
    可合并的 Block 必须属于同一逻辑上下文。

    - 不跨 PDF 页
    - 不跨 Sheet
    - 不跨 PPT Slide
    - 不跨图片
    - 不混合 text/table/transcript/frame OCR 等内容类型

    音视频时间段不在这里比较，因为相邻 transcript 需要继续聚合。
    """
    return (
        a.section_path == b.section_path
        and a.content_type == b.content_type
        and a.page == b.page
        and a.sheet_name == b.sheet_name
        and a.slide_number == b.slide_number
        and a.image_index == b.image_index
    )


def tail_overlap_text(text: str, overlap_chars: int) -> str:
    if overlap_chars <= 0:
        return ""

    return text[-overlap_chars:]


def merge_group(blocks: List[CanonicalBlock], text: str) -> CanonicalBlock:
    ref = blocks[0]

    row_starts = [b.row_start for b in blocks if b.row_start is not None]
    row_ends = [b.row_end for b in blocks if b.row_end is not None]

    media_starts = [
        b.media_start_sec
        for b in blocks
        if b.media_start_sec is not None
    ]
    media_ends = [
        b.media_end_sec
        for b in blocks
        if b.media_end_sec is not None
    ]

    return CanonicalBlock(
        content_type=ref.content_type,
        content=text.strip(),
        section_path=ref.section_path,

        page=ref.page,

        sheet_name=ref.sheet_name,
        row_start=min(row_starts) if row_starts else None,
        row_end=max(row_ends) if row_ends else None,

        slide_number=ref.slide_number,
        image_index=ref.image_index,

        media_start_sec=min(media_starts) if media_starts else None,
        media_end_sec=max(media_ends) if media_ends else None,
    )


def merge_blocks_to_chunks(
    blocks: List[CanonicalBlock],
    max_chars: int,
    overlap_chars: int,
) -> List[CanonicalBlock]:
    expanded: List[CanonicalBlock] = []

    for block in blocks:
        expanded.extend(block_to_units(block, max_chars))

    if not expanded:
        return []

    result: List[CanonicalBlock] = []

    current_blocks: List[CanonicalBlock] = []
    current_text = ""

    def flush():
        nonlocal current_blocks, current_text

        if current_blocks and current_text.strip():
            result.append(
                merge_group(
                    current_blocks,
                    current_text,
                )
            )

        current_blocks = []
        current_text = ""

    for block in expanded:
        if not current_blocks:
            current_blocks = [block]
            current_text = block.content
            continue

        if not same_context(current_blocks[-1], block):
            flush()

            current_blocks = [block]
            current_text = block.content
            continue

        candidate = current_text + "\n\n" + block.content

        if len(candidate) <= max_chars:
            current_blocks.append(block)
            current_text = candidate
            continue

        old_text = current_text
        flush()

        # 这些结构不做字符级 overlap，避免产生残缺表格行/转写片段。
        no_overlap_types = {
            "table",
            "table_row",
            "code",
            "transcript",
            "video_frame_ocr",
            "image_ocr",
        }

        if block.content_type in no_overlap_types:
            current_text = block.content
        else:
            overlap = tail_overlap_text(old_text, overlap_chars)

            current_text = (
                (overlap + "\n\n" + block.content).strip()
                if overlap
                else block.content
            )

        current_blocks = [block]

    flush()
    return result


def build_chunks_for_file(
    path: Path,
    kb_root: Path,
    max_chars: int,
    overlap_chars: int,
) -> List[Chunk]:

    parser = get_parser(path)

    if parser is None:
        raise ValueError(f"不支持的文件格式：{path.suffix}")

    raw_blocks = parser.parse(path)

    relative_path = path.relative_to(kb_root).as_posix()
    document_id = stable_document_id(relative_path)

    domain, subdomain = infer_domain_subdomain(path, kb_root)
    document_type = parser.document_type

    title = path.stem
    source_format = path.suffix.lower().lstrip(".")

    merged = merge_blocks_to_chunks(
        raw_blocks,
        max_chars,
        overlap_chars,
    )

    chunks: List[Chunk] = []

    for chunk_index, block in enumerate(merged):
        content = block.content.strip()

        if len(content) < 8:
            continue

        content_hash = sha1_short(content, 16)
        chunk_id = stable_chunk_id(
            document_id,
            chunk_index,
            content,
        )

        chunks.append(
            Chunk(
                document_id=document_id,
                chunk_id=chunk_id,

                content=content,

                domain=domain,
                subdomain=subdomain,
                document_type=document_type,

                source_file=path.name,
                relative_path=relative_path,
                source_format=source_format,

                title=title,
                section_path=block.section_path or title,

                page=block.page,

                sheet_name=block.sheet_name,
                row_start=block.row_start,
                row_end=block.row_end,

                tags=[],

                chunk_index=chunk_index,
                char_count=len(content),
                content_hash=content_hash,

                content_type=block.content_type,
                slide_number=block.slide_number,
                image_index=block.image_index,

                media_start_sec=block.media_start_sec,
                media_end_sec=block.media_end_sec,
            )
        )

    return chunks


def scan_supported_files(kb_root: Path) -> List[Path]:
    exts = supported_extensions()

    return sorted(
        p
        for p in kb_root.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    )


def run(
    kb_root: Path,
    output_path: Path,
    error_path: Path,
    max_chars: int,
    overlap_chars: int,
):
    if not kb_root.exists():
        raise FileNotFoundError(f"知识库目录不存在：{kb_root}")

    files = scan_supported_files(kb_root)

    print(f"支持格式：{', '.join(sorted(supported_extensions()))}")
    print(f"发现 {len(files)} 个可处理文件。")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    ok_files = 0
    errors = []

    with output_path.open("w", encoding="utf-8") as fout:
        for idx, path in enumerate(files, start=1):
            try:
                chunks = build_chunks_for_file(
                    path=path,
                    kb_root=kb_root,
                    max_chars=max_chars,
                    overlap_chars=overlap_chars,
                )

                for chunk in chunks:
                    fout.write(
                        json.dumps(
                            chunk.to_dict(),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                total_chunks += len(chunks)
                ok_files += 1

                print(
                    f"[{idx}/{len(files)}] OK  "
                    f"{path.relative_to(kb_root)} "
                    f"-> {len(chunks)} chunks"
                )

            except Exception as e:
                item = {
                    "file": str(path),
                    "error": repr(e),
                }

                errors.append(item)

                print(
                    f"[{idx}/{len(files)}] ERR "
                    f"{path.relative_to(kb_root)} "
                    f"-> {e}"
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

    print("\n处理完成")
    print(f"成功文件：{ok_files}")
    print(f"失败文件：{len(errors)}")
    print(f"总 chunks：{total_chunks}")
    print(f"Chunk JSONL：{output_path}")
    print(f"错误日志：{error_path}")


def main():
    parser = argparse.ArgumentParser(
        description="多格式 / 多媒体知识库清洗与切片"
    )

    parser.add_argument("--input", type=Path, default=KB_ROOT)
    parser.add_argument("--output", type=Path, default=CHUNK_JSONL)
    parser.add_argument("--errors", type=Path, default=ERROR_JSONL)
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS)
    parser.add_argument("--overlap-chars", type=int, default=OVERLAP_CHARS)

    args = parser.parse_args()

    run(
        kb_root=args.input,
        output_path=args.output,
        error_path=args.errors,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
    )


if __name__ == "__main__":
    main()
