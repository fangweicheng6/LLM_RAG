# -*- coding: utf-8 -*-
"""
阶段 2：
读取 clean_chunk.py 生成的标准 Chunk JSONL
-> Embedding
-> 按 domain 自动写入 Chroma collection（仅 metadata + embeddings，不传 documents）

这一步不重新解析原始文件。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set


from utils.config_loader import settings
from utils.schema import Chunk
from ingestion import index_state
from ingestion.source_store import SourceResolver
from utils.index_audit import assert_index_collection, validate_storage_paths


def load_chunks(path: Path, only_ids: Set[str] | None = None) -> List[Chunk]:
    """读取索引行并水合正文。

    ``only_ids`` 为 None 时读全部（与旧行为一致）；给定集合时只水合命中的行，
    未命中的行直接跳过——增量场景下绝大多数行都不需要读快照。
    """
    chunks: List[Chunk] = []
    resolver = SourceResolver()

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                if "content" in data or "title" in data or "section_path" in data:
                    raise ValueError("旧版正文 JSONL 不允许入库，请运行 clean_chunk.py 生成新版索引")
                chunk_id = data.pop("chunk_id")
                # 跳过判断放在格式守卫之后：格式检查不解压快照，代价可忽略，
                # 但漏检会让旧版正文悄悄混进向量库。
                if only_ids is not None and chunk_id not in only_ids:
                    continue
                item = resolver.hydrate({"chunk_id": chunk_id, "metadata": data})
                values = dict(item["metadata"])
                values.pop("index_schema_version")
                values["keywords"] = json.loads(values["keywords"])
                for optional in ("page", "sheet_name", "row_start", "row_end"):
                    values.setdefault(optional, None)
                chunks.append(Chunk(chunk_id=chunk_id, content=item["document"], tags=[], **values))
            except Exception as e:
                raise RuntimeError(f"JSONL 第 {line_no} 行解析失败：{e}") from e

    return chunks


_embedding_model = None


def load_embedding_model():
    """离线优先加载 embedding 模型，避免入库阶段访问 HuggingFace Hub 挂起。

    与 pipeline.retrieval.retriever.load_embedding_model 保持一致：
    先 local_files_only=True 读本地缓存，失败再在线降级；全局缓存避免重复加载。
    """
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    import warnings
    # sentence-transformers / transformers 内部仍调用已废弃的 torch.jit.script，
    # 属第三方无害噪音，定向屏蔽，避免每次启动刷屏。
    warnings.filterwarnings(
        "ignore",
        message=r"`torch\.jit\.script` is deprecated",
        category=FutureWarning,
    )

    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Embedding model: {settings.model.embedding.name}")
    print(f"Device: {device}")

    try:
        model = SentenceTransformer(
            settings.model.embedding.name,
            device=device,
            local_files_only=True,
        )
    except Exception as e:
        print(f"[WARN] 本地 embedding cache 加载失败: {e}")
        print("[WARN] 尝试在线加载 embedding model...")
        model = SentenceTransformer(
            settings.model.embedding.name,
            device=device,
        )

    _embedding_model = model
    return model


def build_embedding_text(chunk: Chunk) -> str:
    """
    文档 embedding 的输入文本。

    chunk.content 已烘焙确定性上下文头（CCH，含 文档/章节/工作表/领域），
    这是与查询侧唯一对称的语义信号。文档独有的定位字段（文档类型 / 章节分块序号 /
    页码 / Slide / 媒体时间范围）在查询侧永远不会出现，把它们烘进 doc 向量只会
    向向量注入与问题无关的噪声、稀释正文的语义匹配，故不再拼接，仅使用正文。
    """
    return chunk.content


def get_client():
    import chromadb
    validate_storage_paths(settings.paths.chroma_dir, settings.paths.source_store_dir)
    settings.paths.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(settings.paths.chroma_dir))


def get_collection(client, domain: str):
    domain = domain.lower()

    if domain not in settings.domains.allowed:
        raise ValueError(
            f"不允许创建 domain={domain} 的 collection。"
            f"仅允许：{', '.join(settings.domains.allowed)}"
        )

    name = settings.paths.collection_prefix + domain

    collection = client.get_or_create_collection(
        name=name,
        metadata={
            "description": f"RAG knowledge domain: {domain}",
            "hnsw:space": "cosine",
            # Chroma 1.x 默认 hnsw:sync_threshold=1000：当 collection 的向量数
            # 小于该阈值时，WAL 永远不会同步到 HNSW 索引，导致 link_lists.bin 为空、
            # 检索时报 "Nothing found on disk"。这里调低到 100，确保小 domain
            # （如 pcb=984、schematic=912）也能正常把 HNSW 落盘。
            "hnsw:sync_threshold": 100,
            "hnsw:batch_size": 100,
            "index_schema_version": settings.paths.index_schema_version,
        },
        embedding_function=None,
    )

    assert_index_collection(collection)
    return collection


def replace_domain_documents(collection, chunks: List[Chunk], embeddings):
    """
    同一文档先写新块，成功后删除旧块，避免写入失败时先丢失全部旧记录。
    """
    assert_index_collection(collection, scan=True)
    doc_ids = sorted({c.document_id for c in chunks})
    ids = [c.chunk_id for c in chunks]
    metadatas = [c.chroma_metadata() for c in chunks]
    embedding_list = embeddings.tolist()
    if len(embedding_list) != len(chunks) or len(set(ids)) != len(ids):
        raise ValueError("向量数与索引数不一致，或 chunk_id 重复")
    # 入库边界再次校验，防止调用方绕开 load_chunks 写入无定位的 Chunk。
    resolver = SourceResolver()
    for chunk_id, meta in zip(ids, metadatas):
        resolver.hydrate({"chunk_id": chunk_id, "metadata": meta})

    batch_size = 128

    for start in range(0, len(chunks), batch_size):
        end = start + batch_size

        collection.upsert(
            ids=ids[start:end],
            metadatas=metadatas[start:end],
            embeddings=embedding_list[start:end],
        )

    current_ids = set(ids)
    for doc_id in doc_ids:
        previous = collection.get(where={"document_id": doc_id}, include=["metadatas"])
        stale = [chunk_id for chunk_id in previous["ids"] if chunk_id not in current_ids]
        for start in range(0, len(stale), batch_size):
            collection.delete(ids=stale[start:start + batch_size])


# 索引扫描的唯一实现在 index_state（审计与对账共用，避免逻辑漂移）。
scan_index_ids = index_state.scan_index_ids


def reconcile_collection(
    collection,
    expected_ids: Set[str],
    chunks: List[Chunk],
    embeddings=None,
    batch_size: int = 128,
) -> Dict[str, int]:
    """把 collection 对齐到 expected_ids：只写入缺的，只删除多的。

    与 ``replace_domain_documents`` 的「整域替换」不同，这里**不假设 chunks 是该域
    的全量**，因此可以被增量与全量共用，也让两种模式结果一致（整域替换只清理
    「本批出现过的 document_id」，整份文件被删时其向量永远清不掉）。

    对账天然幂等：任何中断留下的中间状态都会被下一次运行修好。
    """
    ids = [c.chunk_id for c in chunks]
    existing: Set[str] = set(collection.get(include=[])["ids"])
    stale = sorted(existing - expected_ids)

    # 无写入则跳过整库扫描式审计：它要分页读回全部 metadata，代价不小，而这里并不落盘。
    if not ids and not stale:
        return {"added": 0, "deleted": 0}

    assert_index_collection(collection, scan=True)

    written = set(ids)
    missing = expected_ids - existing

    # 允许「多写」——--rebuild / 换 embedding 模型时会连已在库里的 chunk 一起重写以刷新
    # 向量；但绝不允许少写（缺的没补上）或写出索引里没有的 id。
    if len(written) != len(ids):
        raise ValueError(f"待写入的 chunk 集合存在重复 id（{len(ids) - len(written)} 个）")
    if not missing <= written:
        raise ValueError(
            "待写入的 chunk 集合不足以补齐索引期望："
            f"缺少 {len(missing - written)} 个（应补 {len(missing)} 个，实得 {len(written)} 个）"
        )
    if not written <= expected_ids:
        raise ValueError(
            f"待写入的 chunk 集合包含索引中不存在的 id（{len(written - expected_ids)} 个）"
        )

    metadatas = [c.chroma_metadata() for c in chunks]

    if ids:
        embedding_list = embeddings.tolist()
        if len(embedding_list) != len(chunks):
            raise ValueError("向量数与索引数不一致")
        # 入库边界再次校验，防止调用方绕开 load_chunks 写入无定位的 Chunk。
        resolver = SourceResolver()
        for chunk_id, meta in zip(ids, metadatas):
            resolver.hydrate({"chunk_id": chunk_id, "metadata": meta})

    # 先写后删：写入失败时不至于先丢掉旧记录。
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings.tolist()[start:end],
        )

    for start in range(0, len(stale), batch_size):
        collection.delete(ids=stale[start:start + batch_size])

    return {"added": len(ids), "deleted": len(stale)}


def _domains_to_touch(client, by_domain: Dict[str, Set[str]]) -> List[str]:
    """需要打开并处理 collection 的 domain。

    除了索引里出现的 domain，还要带上「Chroma 里已经存在、但索引里已经没有」的
    domain——该域的文件可能被全部删除，此时索引里一个 id 都没有，恰恰最需要去把
    残留向量删掉。
    """
    prefix = settings.paths.collection_prefix
    existing = {
        item if isinstance(item, str) else item.name
        for item in client.list_collections()
    }
    domains = {
        name[len(prefix):]
        for name in existing
        if name.startswith(prefix)
    }
    domains |= set(by_domain)
    return sorted(domains)


def _embedding_model_changed(input_path: Path) -> bool:
    """索引 sidecar 记录的嵌入模型与当前配置不一致时，强制全量重嵌。

    sidecar 只是 advisory：缺失时无法判断存量向量是谁产出的，此时只告警不阻断
    （阻断会让第一次运行被迫全量重嵌），需要重建时用 --rebuild 显式声明。
    """
    remembered = index_state.read_state(index_state.state_path(input_path)).get("embedding_model")
    current = settings.model.embedding.name

    if remembered is None:
        print(
            "[WARN] 状态文件缺少 embedding 模型记录，无法确认存量向量由谁产出；"
            "如需强制重嵌请加 --rebuild"
        )
        return False
    if remembered != current:
        print(f"[WARN] embedding 模型已从 {remembered} 变为 {current}，本次强制全部重嵌")
        return True
    return False


def _remember_embedding(input_path: Path) -> None:
    try:
        index_state.write_state(
            index_state.state_path(input_path),
            {"embedding_model": settings.model.embedding.name},
        )
    except OSError as exc:
        print(f"[WARN] 入库状态文件写入失败（不影响本次结果）：{exc}")


def run(input_path: Path, *, rebuild: bool = False) -> Dict[str, Dict[str, int]]:
    """把向量库对账到索引。

    ``rebuild=True`` 时忽略 Chroma 现有内容，强制把索引里的每个 chunk 重新嵌入。
    """
    if not input_path.exists():
        raise FileNotFoundError(
            f"找不到 Chunk JSONL：{input_path}\n"
            "请先运行：python clean_chunk.py"
        )

    by_domain = scan_index_ids(input_path)
    if not by_domain:
        raise RuntimeError("Chunk JSONL 为空。")

    invalid_domains = sorted(
        domain for domain in by_domain if domain not in settings.domains.allowed
    )

    if invalid_domains:
        raise RuntimeError(
            "Chunk JSONL 中存在不允许的一级 domain："
            + ", ".join(invalid_domains)
            + "。仅允许："
            + ", ".join(settings.domains.allowed)
        )

    total = sum(len(ids) for ids in by_domain.values())
    print(f"读取索引：{total} chunks")
    print(f"domain 数量：{len(by_domain)}")

    client = get_client()

    # 如果旧版本数据库中已经存在其他 rag_* Collection，
    # 不静默删除用户数据，而是直接报错，要求先清理旧库再重新入库。
    existing_collection_names = {
        item if isinstance(item, str) else item.name
        for item in client.list_collections()
    }

    allowed_collection_names = {
        settings.paths.collection_prefix + domain
        for domain in settings.domains.allowed
    }

    illegal_collections = sorted(
        name
        for name in existing_collection_names
        if name.startswith(settings.paths.collection_prefix)
        and name not in allowed_collection_names
    )

    if illegal_collections:
        raise RuntimeError(
            "Chroma 中存在旧的非法 RAG Collection："
            + ", ".join(illegal_collections)
            + "。当前版本只允许："
            + ", ".join(sorted(allowed_collection_names))
            + "。请备份后清理旧 Chroma 目录，再重新运行 python ingest.py。"
        )

    # 在任何写入之前检查已有集合；旧数据库不能靠 upsert 擦除历史正文。
    for name in existing_collection_names & allowed_collection_names:
        assert_index_collection(client.get_collection(name=name, embedding_function=None), scan=True)

    force_rebuild = rebuild or _embedding_model_changed(input_path)
    model = None
    summary: Dict[str, Dict[str, int]] = {}

    for domain in _domains_to_touch(client, by_domain):
        expected = by_domain.get(domain, set())
        collection = get_collection(client, domain)
        existing: Set[str] = set(collection.get(include=[])["ids"])

        missing = expected if force_rebuild else (expected - existing)

        if not missing and not (existing - expected):
            print(f"\n[{domain}] 无需变更（{len(expected)} chunks）")
            summary[domain] = {"added": 0, "deleted": 0}
            continue

        chunks: List[Chunk] = []
        embeddings = None

        if missing:
            chunks = load_chunks(input_path, only_ids=missing)
            loaded = {c.chunk_id for c in chunks}
            if loaded != missing:
                raise RuntimeError(
                    f"[{domain}] 索引中有 {len(missing - loaded)} 个 chunk 读不出来"
                    "（行缺失或 chunk_id 重复），请重新运行切片阶段"
                )
            if model is None:
                model = load_embedding_model()
            embeddings = model.encode(
                [build_embedding_text(c) for c in chunks],
                batch_size=settings.ingestion.embedding.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

        result = reconcile_collection(collection, expected, chunks, embeddings)
        summary[domain] = result
        print(
            f"\n[{domain}] 写入 {result['added']}，删除 {result['deleted']}；"
            f"当前 count={collection.count()}"
        )

    # 只在本次真的执行过嵌入时才记录模型名：否则一次空转的对账会把「存量向量出处
    # 未知」洗白成「存量向量 = 当前配置模型」，此后再也发现不了模型不一致。
    if model is not None:
        _remember_embedding(input_path)
    print(f"\n向量库对账完成：{settings.paths.chroma_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Chunk Embedding + Chroma 入库（对账式）")
    parser.add_argument("--input", type=Path, default=settings.paths.chunk_jsonl)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="忽略向量库现有内容，强制把索引里的每个 chunk 重新嵌入",
    )
    args = parser.parse_args()

    run(args.input, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
