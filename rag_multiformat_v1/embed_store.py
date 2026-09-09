# -*- coding: utf-8 -*-
"""
阶段 2：
读取 clean_chunk.py 生成的标准 Chunk JSONL
-> Embedding
-> 按 domain 自动写入 Chroma collection

这一步不重新解析原始文件。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import chromadb
import torch
from sentence_transformers import SentenceTransformer

from config import (
    CHUNK_JSONL,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    EMBED_BATCH_SIZE,
    COLLECTION_PREFIX,
)
from schema import Chunk


def load_chunks(path: Path) -> List[Chunk]:
    chunks: List[Chunk] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                chunks.append(Chunk(**data))
            except Exception as e:
                raise RuntimeError(f"JSONL 第 {line_no} 行解析失败：{e}") from e

    return chunks


def load_embedding_model() -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Device: {device}")

    return SentenceTransformer(
        EMBEDDING_MODEL,
        device=device,
    )


def build_embedding_text(chunk: Chunk) -> str:
    """
    文档 embedding 时把结构化上下文一起编码。
    多媒体内容会加入 content_type / 页码 / Sheet / Slide / 时间范围。
    """
    context_parts = [
        f"知识域：{chunk.domain}",
        f"子域：{chunk.subdomain}",
        f"文档类型：{chunk.document_type}",
        # f"内容类型：{chunk.content_type}",
        f"文档：{chunk.title}",
        f"章节：{chunk.section_path}",
    ]

    if chunk.page is not None:
        context_parts.append(f"页码：{chunk.page}")

    if chunk.sheet_name:
        context_parts.append(f"Sheet：{chunk.sheet_name}")

    if chunk.slide_number is not None:
        context_parts.append(f"Slide：{chunk.slide_number}")

    if chunk.media_start_sec is not None:
        if chunk.media_end_sec is not None:
            context_parts.append(
                f"媒体时间：{chunk.media_start_sec:.1f}-{chunk.media_end_sec:.1f}s"
            )
        else:
            context_parts.append(f"媒体时间：{chunk.media_start_sec:.1f}s")

    return "\n".join(context_parts) + "\n\n" + chunk.content


def get_client():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(client, domain: str):
    name = COLLECTION_PREFIX + domain.lower()

    return client.get_or_create_collection(
        name=name,
        metadata={
            "description": f"RAG knowledge domain: {domain}",
            "hnsw:space": "cosine",
        },
        embedding_function=None,
    )


def replace_domain_documents(collection, chunks: List[Chunk], embeddings):
    """
    可重复执行：
    同一 document_id 先删旧 chunks，再写新 chunks。
    """
    doc_ids = sorted({c.document_id for c in chunks})

    for doc_id in doc_ids:
        try:
            collection.delete(where={"document_id": doc_id})
        except Exception:
            pass

    ids = [c.chunk_id for c in chunks]
    documents = [c.content for c in chunks]
    metadatas = [c.chroma_metadata() for c in chunks]
    embedding_list = embeddings.tolist()

    batch_size = 128

    for start in range(0, len(chunks), batch_size):
        end = start + batch_size

        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embedding_list[start:end],
        )


def run(input_path: Path):
    if not input_path.exists():
        raise FileNotFoundError(
            f"找不到 Chunk JSONL：{input_path}\n"
            "请先运行：python clean_chunk.py"
        )

    chunks = load_chunks(input_path)
    if not chunks:
        raise RuntimeError("Chunk JSONL 为空。")

    grouped: Dict[str, List[Chunk]] = defaultdict(list)

    for chunk in chunks:
        grouped[chunk.domain].append(chunk)

    print(f"读取 chunks：{len(chunks)}")
    print(f"domain 数量：{len(grouped)}")

    model = load_embedding_model()
    client = get_client()

    for domain, domain_chunks in sorted(grouped.items()):
        print(f"\n[{domain}] {len(domain_chunks)} chunks")

        texts = [build_embedding_text(c) for c in domain_chunks]

        embeddings = model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        collection = get_collection(client, domain)
        replace_domain_documents(collection, domain_chunks, embeddings)

        print(
            f"写入 collection={COLLECTION_PREFIX + domain.lower()}，"
            f"当前 count={collection.count()}"
        )

    print(f"\nEmbedding 与入库完成：{CHROMA_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Chunk Embedding + Chroma 入库")
    parser.add_argument("--input", type=Path, default=CHUNK_JSONL)
    args = parser.parse_args()

    run(args.input)


if __name__ == "__main__":
    main()
