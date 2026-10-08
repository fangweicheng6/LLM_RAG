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
from typing import Dict, List


from utils.config_loader import settings
from utils.schema import Chunk
from ingestion.source_store import SourceResolver
from utils.index_audit import assert_index_collection, validate_storage_paths


def load_chunks(path: Path) -> List[Chunk]:
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


def run(input_path: Path):
    if not input_path.exists():
        raise FileNotFoundError(
            f"找不到 Chunk JSONL：{input_path}\n"
            "请先运行：python clean_chunk.py"
        )

    chunks = load_chunks(input_path)
    if not chunks:
        raise RuntimeError("Chunk JSONL 为空。")

    invalid_domains = sorted(
        {
            chunk.domain
            for chunk in chunks
            if chunk.domain not in settings.domains.allowed
        }
    )

    if invalid_domains:
        raise RuntimeError(
            "Chunk JSONL 中存在不允许的一级 domain："
            + ", ".join(invalid_domains)
            + "。仅允许："
            + ", ".join(settings.domains.allowed)
        )

    grouped: Dict[str, List[Chunk]] = defaultdict(list)

    for chunk in chunks:
        grouped[chunk.domain].append(chunk)

    print(f"读取 chunks：{len(chunks)}")
    print(f"domain 数量：{len(grouped)}")

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
    model = load_embedding_model()

    for domain, domain_chunks in sorted(grouped.items()):
        print(f"\n[{domain}] {len(domain_chunks)} chunks")

        texts = [build_embedding_text(c) for c in domain_chunks]

        embeddings = model.encode(
            texts,
            batch_size=settings.ingestion.embedding.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        collection = get_collection(client, domain)
        replace_domain_documents(collection, domain_chunks, embeddings)

        print(
            f"写入 collection={settings.paths.collection_prefix + domain.lower()}，"
            f"当前 count={collection.count()}"
        )

    print(f"\nEmbedding 与入库完成：{settings.paths.chroma_dir}")


def main():
    parser = argparse.ArgumentParser(description="Chunk Embedding + Chroma 入库")
    parser.add_argument("--input", type=Path, default=settings.paths.chunk_jsonl)
    args = parser.parse_args()

    run(args.input)


if __name__ == "__main__":
    main()
