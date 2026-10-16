# -*- coding: utf-8 -*-
"""入库状态 sidecar 与增量分类逻辑。

本模块承担两件事：

1. **advisory sidecar**（``<索引>.state.json``）：记录切片流水线指纹、产出为空的文件名单，
   以及最近一次运行的摘要。它随时可以删掉——任何缺失、损坏或不一致都只会退化成
   「按全量处理」，绝不会产出坏索引。**它不是增量的事实源。**

2. **单遍扫描索引 JSONL 得出分类**：``source_file_hash`` / ``relative_path`` /
   ``document_id`` / ``domain`` 都写在索引的每一行里，因此「哪些文件未变、哪些变了、
   哪些新增、哪些已删除」可以直接从索引推导，不需要额外的账本文件。
   多一份账本就多一个「谁先提交」的失效面，得不偿失。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from utils.config_loader import settings

# 切片口径的版本号。修改任何影响 chunk 产出的逻辑（切片规则、清洗规则、上下文头、
# 解析器选择）都必须手动 +1，否则增量会把新旧两种口径的切片混在一份索引里。
CHUNK_PIPELINE_VERSION = 1

STATE_SUFFIX = ".state.json"

_STEM_WS_RE = re.compile(r"\s+")


def state_path(output_path: Path) -> Path:
    """sidecar 路径从索引路径派生，不新增 settings.paths 键。

    这样测试里 ``patch.object(settings.paths, "chunk_jsonl", tmp)`` 会自动连带改掉
    sidecar 落点；若单独加一个 path 配置键，漏 patch 时会写进真实索引目录。
    """
    return Path(output_path).parent / (Path(output_path).name + STATE_SUFFIX)


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def fingerprint_parts() -> dict:
    """指纹的全部输入项（可读，用于诊断「到底什么变了」）。"""
    ingestion = dict(settings.ingestion.as_dict())
    # 排除只影响写入/编排、不影响 chunk 产出的配置，免得调个批大小或删除阈值就触发
    # 一次全量重切。默认「全含」而不是「白名单」：将来新增的切片配置若忘了登记，
    # 安全的方向是误触发一次全量，而不是悄悄产出混合口径的索引。
    for key in ("embedding", "incremental"):
        ingestion.pop(key, None)

    return {
        "chunk_pipeline_version": CHUNK_PIPELINE_VERSION,
        "index_schema_version": settings.paths.index_schema_version,
        "ingestion": ingestion,
        # 切片用的 tokenizer 直接取自 embedding 模型（docling_chunker 用
        # AutoTokenizer.from_pretrained(settings.model.embedding.name) 喂 HybridChunker），
        # 它决定 chunk 的 token 预算与切分边界，必须进指纹。
        "chunking_tokenizer": settings.model.embedding.name,
        # 每个包都实际参与正文产出：docling/rapidocr 解析文档与图像，
        # openpyxl 读 xlsx/csv，faster-whisper 做 ASR，pymupdf 抽 PDF 文本。
        "parsers": {
            name: _package_version(name)
            for name in (
                "docling",
                "docling-core",
                "rapidocr-onnxruntime",
                "openpyxl",
                "faster-whisper",
                "pymupdf",
            )
        },
    }


def hash_parts(parts: dict) -> str:
    return hashlib.sha256(
        json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def pipeline_fingerprint() -> str:
    """切片口径指纹：这个索引是否由同一套切片逻辑产出的。

    文件哈希只能回答「文件字节变了吗」，回答不了「解析器、切片参数或 tokenizer 变了吗」。
    指纹不一致时增量必须拒绝，否则会静默产出新旧口径混合的语料。
    """
    return hash_parts(fingerprint_parts())


def fingerprint_diff(stored: dict, current: dict) -> List[str]:
    """返回**真正变了的输入项**。

    只看两边都有的键：指纹的定义本身也会演进（本次就新增了 tokenizer 与若干解析器
    版本），这种「键的增删」不代表语料口径变了，不该误判成需要全量重切。只有同名键的
    取值不同才是真的变了。
    """
    if not isinstance(stored, dict):
        return ["<缺少可比的指纹输入项>"]
    return sorted(key for key in set(stored) & set(current) if stored[key] != current[key])


def read_state(path: Path) -> dict:
    """读取 sidecar；缺失或损坏一律返回空 dict，永不抛。"""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_state(path: Path, updates: dict) -> None:
    """读-合并-写 sidecar，同目录临时文件 + 原子替换。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = read_state(path)
    data.update(updates)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def normalized_stem(relative_path: str) -> str:
    """跨文件去重的「孪生」分组键。

    每条 chunk 的正文都以确定性上下文头（CCH）开头：``文档《{title}》…``，其中 title 是
    ``stem.replace("_", " ")``（见 clean_chunk._row_context_header 与
    docling_chunker._docling_context_header）。跨文件去重指纹是对整段正文归一化后算的，
    所以**两个不同文件的 chunk 指纹相等，必然意味着它们的归一化 stem 相等**。
    于是「同 stem 组」就是所有可能互相挤掉内容的文件集合，且这是充分覆盖的。

    关键：这里的归一化必须**与指纹侧同口径**。``content_fingerprint`` 会先删 BOM 再做 NFC
    归一化（chunk_clean.normalize_text），若分组键不做同样处理，就会出现「指纹判为同一块、
    分组却判为不同孪生」的错配——被挤掉的块将永久无法复活。NFD/NFC 与 BOM 在跨平台拷贝
    （macOS/Linux 产出 NFD，Windows 另存为 NFC）时是现实存在的。
    """
    stem = unicodedata.normalize("NFC", Path(relative_path).stem.replace("﻿", ""))
    return _STEM_WS_RE.sub(" ", stem.replace("_", " ")).strip().lower()


@dataclass
class IndexSnapshot:
    """索引 JSONL 的单遍扫描结果。只读定位字段，**不做水合**。"""

    paths: Dict[str, str] = field(default_factory=dict)       # relative_path -> source_file_hash
    lines: Dict[str, List[str]] = field(default_factory=dict)  # relative_path -> 原始行文本
    order: List[str] = field(default_factory=list)             # relative_path 首次出现的顺序
    chunk_ids: set = field(default_factory=set)
    raw_count: int = 0
    conflicts: List[str] = field(default_factory=list)   # 同一路径出现多个文件哈希
    duplicates: List[str] = field(default_factory=list)  # 重复 chunk_id
    schema_mismatch: bool = False
    tail_truncated: bool = False  # 尾部有读不成 JSON 的残行（写入被中断）


def scan_index(
    jsonl_path: Path,
    *,
    drop_last_file: bool = False,
    max_duplicate_reports: int = 5,
) -> IndexSnapshot:
    """单遍流式扫描索引，取分类所需的定位字段并保留原始行文本。

    保留原始行是为了让未变文件的行能被**逐字节**写回：索引行是
    ``{"chunk_id": ...} + chroma_metadata()`` 的序列化结果，重新序列化会改变键序、
    数值格式，也会让整份索引在下游 diff 里显得全变。这里连 json 都不重新 dump。

    ``drop_last_file=True`` 用于「中断的 .tmp」：一个文件的行走缓冲写出，原生崩溃时
    最后一个文件的记录可能只落了一半，把最后一组整组丢弃重做。
    """
    snapshot = IndexSnapshot()
    path = Path(jsonl_path)
    if not path.exists():
        return snapshot

    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.rstrip("\n")
            if not stripped.strip():
                continue
            try:
                record = json.loads(stripped)
            except ValueError:
                # 写入被中断时最后一行可能是半条记录。行是按顺序写的，出现残行说明后面
                # 也不可信，到此为止；调用方决定是「丢弃尾部重做」还是「直接报错」。
                snapshot.tail_truncated = True
                break
            snapshot.raw_count += 1

            chunk_id = record["chunk_id"]
            if chunk_id in snapshot.chunk_ids:
                if len(snapshot.duplicates) < max_duplicate_reports:
                    snapshot.duplicates.append(chunk_id)
            snapshot.chunk_ids.add(chunk_id)

            if record.get("index_schema_version") != settings.paths.index_schema_version:
                snapshot.schema_mismatch = True

            relative = record["relative_path"]
            file_hash = record.get("source_file_hash", "")
            known = snapshot.paths.get(relative)
            if known is None:
                snapshot.paths[relative] = file_hash
                snapshot.order.append(relative)
            elif known != file_hash:
                # 同一文件的行必须共享一个哈希；出现分歧说明索引本身是坏的中间态，
                # 该文件一律按「已变化」处理，强制重切。
                if relative not in snapshot.conflicts:
                    snapshot.conflicts.append(relative)
                snapshot.paths[relative] = file_hash

            snapshot.lines.setdefault(relative, []).append(stripped)

    if drop_last_file and snapshot.order:
        dropped = snapshot.order.pop()
        snapshot.paths.pop(dropped, None)
        # 丢弃的行要连 chunk_id 一起回收，否则会被当成「重复 chunk_id」误报。
        for line in snapshot.lines.pop(dropped, []):
            snapshot.raw_count -= 1
            snapshot.chunk_ids.discard(json.loads(line)["chunk_id"])

    return snapshot


@dataclass
class Plan:
    """增量切片计划。"""

    unchanged: List[Tuple[Path, List[str]]] = field(default_factory=list)
    reparse: List[Path] = field(default_factory=list)
    deleted_paths: List[str] = field(default_factory=list)
    added: List[str] = field(default_factory=list)
    changed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.reparse and not self.deleted_paths


def _stem_groups(kb_root: Path, files: List[Path], snapshot: IndexSnapshot) -> Dict[str, set]:
    """按归一化 stem 分组，成员 = 磁盘上的文件 ∪ 索引里的路径。

    索引路径必须参与分组：否则「A 被删掉」时 B 会单独成组，看不出它曾经有个孪生。
    """
    groups: Dict[str, set] = defaultdict(set)
    for relative in snapshot.paths:
        groups[normalized_stem(relative)].add(relative)
    for path in files:
        groups[normalized_stem(path.relative_to(kb_root).as_posix())].add(
            path.relative_to(kb_root).as_posix()
        )
    return groups


def classify(
    kb_root: Path,
    files: List[Path],
    snapshot: IndexSnapshot,
    *,
    force_reparse_paths: set | None = None,
) -> Plan:
    """按文件哈希把当前磁盘文件分成「未变 / 需重切」，并算出已删除的路径。

    ``files`` 必须是已按路径排序的列表，这样未变文件与新切片文件的输出顺序与全量
    重跑一致。
    """
    kb_root = Path(kb_root)
    plan = Plan()
    force = set(force_reparse_paths or ())
    current_paths = set()

    for path in files:
        relative = path.relative_to(kb_root).as_posix()
        current_paths.add(relative)

        known_hash = snapshot.paths.get(relative)
        if known_hash is None:
            plan.added.append(relative)
            plan.reparse.append(path)
            continue

        try:
            from ingestion.source_store import file_digest

            current_hash = file_digest(path)
        except OSError as exc:
            plan.warnings.append(f"{relative}: 无法计算文件哈希（{exc}），按已变化处理")
            plan.changed.append(relative)
            plan.reparse.append(path)
            continue

        if current_hash != known_hash or relative in snapshot.conflicts or relative in force:
            plan.changed.append(relative)
            plan.reparse.append(path)
        else:
            plan.unchanged.append((path, snapshot.lines.get(relative, [])))

    plan.deleted_paths = sorted(set(snapshot.paths) - current_paths)

    # 孪生扩张：同 stem 组里只要有一个成员发生增/删/改，整组（仍存在的成员）都重切。
    # 否则被组内更早文件挤掉的重复块，在「赢家被删」时永远无法复活。
    dirty_stems = {
        normalized_stem(item)
        for item in (*plan.added, *plan.changed, *plan.deleted_paths)
    }
    already = {path.relative_to(kb_root).as_posix() for path in plan.reparse}
    if dirty_stems:
        groups = _stem_groups(kb_root, files, snapshot)
        promoted: set = set()
        for stem in dirty_stems:
            members = groups.get(stem, set())
            if len(members) < 2:
                continue  # 单成员组不可能发生跨文件碰撞
            promoted |= members

        if promoted:
            promoted &= current_paths - already
            if promoted:
                kept = []
                for path, lines in plan.unchanged:
                    relative = path.relative_to(kb_root).as_posix()
                    if relative in promoted:
                        plan.reparse.append(path)
                        plan.changed.append(relative)
                    else:
                        kept.append((path, lines))
                plan.unchanged = kept
                plan.reparse.sort(key=lambda item: item.relative_to(kb_root).as_posix())
                plan.changed.sort()

    plan.warnings.extend(
        f"{relative}: 索引中该文件的行存在多个 source_file_hash，按已变化处理"
        for relative in snapshot.conflicts
    )
    return plan


def scan_index_ids(jsonl_path: Path) -> Dict[str, set]:
    """单遍扫描索引，得到 domain -> {chunk_id}，**不做水合**。

    chunk_id 必须全局唯一、domain 必须合法，否则「期望集合」本身就是错的，
    据此对账会删掉不该删的数据；口径与 evaluation.eval_common.load_source_chunks
    保持一致。
    """
    by_domain: Dict[str, set] = defaultdict(set)
    seen: set = set()

    with Path(jsonl_path).open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                chunk_id = data["chunk_id"]
                domain = data["domain"]
            except (ValueError, KeyError) as exc:
                raise RuntimeError(f"索引 JSONL 第 {line_no} 行无法解析：{exc}") from exc
            if data.get("index_schema_version") != settings.paths.index_schema_version:
                raise RuntimeError(
                    f"索引 JSONL 第 {line_no} 行不是当前 index_schema_version，请重新入库"
                )
            if chunk_id in seen:
                raise RuntimeError(f"索引 JSONL 存在重复 chunk_id：{chunk_id}")
            seen.add(chunk_id)
            by_domain[str(domain)].add(chunk_id)

    return by_domain


def cch_invariant_holds(chunks) -> bool:
    """校验「每条 chunk 都以 CCH 上下文头开头」这个不变量。

    孪生分组的充分性依赖它（见 ``normalized_stem`` 的说明）。一旦不成立，
    增量必须降级为全量，否则跨文件去重可能跨 stem 碰撞而无人察觉。
    """
    return all(chunk.content.startswith("文档《") for chunk in chunks)
