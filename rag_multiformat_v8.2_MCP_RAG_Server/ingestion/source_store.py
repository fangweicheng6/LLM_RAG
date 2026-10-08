"""正文仅保存在受信任文件目录；Chroma/JSONL 只持有定位信息。"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from bisect import bisect_right
from collections import Counter
from dataclasses import replace
from pathlib import Path

from utils.config_loader import settings
from utils.index_audit import validate_storage_paths


class SourceResolutionError(RuntimeError):
    pass


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def extract_keywords(text: str, limit: int = 12) -> list[str]:
    """确定性关键词：英文技术词 + 中文二字词频；不把整句充当关键词。"""
    stop = {"the", "and", "for", "with", "this", "that", "from", "的电", "以及", "可以", "进行", "一个", "通过", "使用"}
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_+.-]{1,23}", text.lower())
    for phrase in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.extend(phrase[i:i + 2] for i in range(len(phrase) - 1))
    counts = Counter(t for t in tokens if t not in stop)
    return [token for token, _ in counts.most_common(limit)]


def located_blocks(blocks, title: str, max_chars: int):
    """按真实页/幻灯片/工作表形成解析文本单元；按章节、位置边界切块。

    行列从 1 开始，char_start/end 为单元内 Python 字符偏移，左闭右开。
    不去重、不改写句子；跨页必须切开，长行用列号精确定位。
    """
    if max_chars <= 0:
        raise ValueError("section_max_chars 必须大于 0")
    units, unit_keys, groups = {}, {}, []
    for block in blocks:
        text = block.content.replace("\r\n", "\n").replace("\r", "\n")
        if not text.strip():
            continue
        key = (block.page, block.slide_number, block.sheet_name)
        unit_id = unit_keys.setdefault(key, str(len(unit_keys) + 1))
        previous = units.get(unit_id, "")
        start = len(previous) + (1 if previous else 0)
        units[unit_id] = previous + ("\n" if previous else "") + text
        section = block.section_path or title
        boundary = (unit_id, section, block.content_type, block.row_start,
                    block.row_end, block.image_index, block.media_start_sec,
                    block.media_end_sec, block.page_end)
        if groups and groups[-1][0] == boundary:
            groups[-1][3] = start + len(text)
        else:
            groups.append([boundary, replace(block, section_path=section), start, start + len(text)])

    slices = []
    for boundary, block, start, end in groups:
        unit_id = boundary[0]
        text = units[unit_id]
        line_starts = [0] + [i + 1 for i, c in enumerate(text) if c == "\n"]
        while start < end:
            stop = min(start + max_chars, end)
            if stop < end:
                newline = text.rfind("\n", start, stop)
                if newline > start:
                    stop = newline + 1
            content = text[start:stop]
            if content.strip():
                first = bisect_right(line_starts, start)
                last = bisect_right(line_starts, stop - 1)
                slices.append((replace(block, content=content), {
                    "source_unit": unit_id,
                    "line_start": first, "line_end": last,
                    "column_start": start - line_starts[first - 1] + 1,
                    "column_end": stop - line_starts[last - 1],
                    "char_start": start, "char_end": stop,
                    "line_basis": "parsed_text_v1",
                }))
            start = stop
    return units, slices


def save_snapshot(chunks, units, root: Path | None = None) -> None:
    if not chunks:
        return

    # source_store 的磁盘格式要求 units 为 {source_unit: parsed_text}。
    # 这里提前校验，避免错误快照一直到 Embedding 阶段才以
    # “list indices must be integers”之类的二次异常暴露。
    if not isinstance(units, dict):
        raise TypeError("source snapshot units 必须是 dict[str, str]")

    for chunk in chunks:
        unit_id = str(chunk.source_unit)
        if unit_id not in units:
            raise ValueError(f"{chunk.chunk_id}: source_unit={unit_id!r} 不存在于 snapshot units")
        unit_text = units[unit_id]
        if not isinstance(unit_text, str):
            raise TypeError(f"{chunk.chunk_id}: snapshot unit 必须是字符串")
        if not isinstance(chunk.char_start, int) or not isinstance(chunk.char_end, int):
            raise ValueError(f"{chunk.chunk_id}: char_start/char_end 必须是整数")
        if not (0 <= chunk.char_start <= chunk.char_end <= len(unit_text)):
            raise ValueError(f"{chunk.chunk_id}: source 字符定位越界")
        recovered = unit_text[chunk.char_start:chunk.char_end]
        if recovered != chunk.content:
            raise ValueError(f"{chunk.chunk_id}: source 定位无法精确恢复 chunk 正文")
        if digest(recovered) != chunk.content_hash or len(recovered) != chunk.char_count:
            raise ValueError(f"{chunk.chunk_id}: source 定位后的正文校验失败")

    root = Path(root if root is not None else settings.paths.source_store_dir).resolve()
    validate_storage_paths(settings.paths.chroma_dir, root)
    entries = {}
    for chunk in chunks:
        index = chunk.to_index_dict()
        index.pop("source_ref", None)
        entries[chunk.chunk_id] = {
            "index": index,
            "title": chunk.title,
            "section_path": chunk.section_path,
        }
    payload = json.dumps({"units": units, "chunks": entries}, ensure_ascii=False,
                         sort_keys=True, separators=(",", ":"))
    source_ref = digest(payload)
    directory = root / source_ref[:2]
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (source_ref + ".json")
    # 临时文件与目标同目录，原子发布；中断不会留下半个有效快照。
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                     dir=directory, delete=False) as stream:
        temp_path = Path(stream.name)
        stream.write(payload)
    try:
        os.replace(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)
    for chunk in chunks:
        chunk.source_ref = source_ref


class SourceResolver:
    """缓存只存活于一次候选批次；不把读取结果写回数据库。"""
    def __init__(self, root: Path | None = None):
        self.root = Path(root if root is not None else settings.paths.source_store_dir).resolve()
        self._snapshots = {}

    def hydrate(self, item: dict) -> dict:
        meta = dict(item.get("metadata") or {})
        ref = meta.get("source_ref", "")
        chunk_id = item.get("chunk_id", "")
        if meta.get("index_schema_version") != settings.paths.index_schema_version or not re.fullmatch(r"[0-9a-f]{64}", ref):
            raise SourceResolutionError(f"{chunk_id}: 不是新版无正文索引，请重新运行 ingest.py")
        if ref not in self._snapshots:
            path = (self.root / ref[:2] / (ref + ".json")).resolve()
            if not path.is_relative_to(self.root):
                raise SourceResolutionError("原文路径越界")
            try:
                payload = path.read_text(encoding="utf-8")
                if digest(payload) != ref:
                    raise ValueError("快照校验失败")
                self._snapshots[ref] = json.loads(payload)
            except (OSError, ValueError) as exc:
                raise SourceResolutionError(f"{chunk_id}: 原文快照缺失或损坏，请恢复 settings.paths.source_store_dir 或重新入库") from exc
        snapshot = self._snapshots[ref]
        try:
            entry = snapshot["chunks"][chunk_id]
            # 索引与外部快照必须完全吻合，防止错页/错域/串文档。
            for key, value in entry["index"].items():
                if (chunk_id if key == "chunk_id" else meta.get(key)) != value:
                    raise ValueError(f"定位字段不匹配: {key}")
            text = snapshot["units"][str(meta["source_unit"])][meta["char_start"]:meta["char_end"]]
            if digest(text) != meta["content_hash"] or len(text) != meta["char_count"]:
                raise ValueError("内容校验失败")
        except (KeyError, TypeError, ValueError) as exc:
            raise SourceResolutionError(f"{chunk_id}: 索引与原文不一致，请重新入库") from exc
        meta.update(title=entry["title"], section_path=entry["section_path"])
        return {**item, "document": text, "metadata": meta}


def hydrate_items(items: list[dict], root: Path | None = None) -> list[dict]:
    resolver = SourceResolver(root)
    return [resolver.hydrate(item) for item in items]


def format_location(meta: dict) -> str:
    parts = [str(meta.get("domain", "")), str(meta.get("relative_path", ""))]
    for key, label in (("page", "page"), ("slide_number", "slide"), ("sheet_name", "sheet")):
        if meta.get(key) is not None:
            parts.append(f"{label}={meta[key]}")
    if meta.get("row_start") is not None:
        parts.append(f"rows={meta['row_start']}-{meta.get('row_end')}")
    if meta.get("media_start_sec") is not None:
        parts.append(f"time={meta['media_start_sec']}-{meta.get('media_end_sec')}s")
    parts.append(f"parsed_unit={meta.get('source_unit')} lines={meta.get('line_start')}-{meta.get('line_end')} "
                 f"columns={meta.get('column_start')}-{meta.get('column_end')}")
    return " | ".join(part for part in parts if part)
