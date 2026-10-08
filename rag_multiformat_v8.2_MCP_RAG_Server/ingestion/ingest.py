# -*- coding: utf-8 -*-
"""
RAG 一键入库入口

执行：
    python ingest.py

自动完成：
1. 扫描知识库
2. 多格式解析
3. 清洗
4. 按章节切片
5. 输出无正文索引 JSONL
6. Embedding
7. 按 topology / device / pcb / control / schematic 写入 Chroma
"""

from __future__ import annotations

from ingestion.clean_chunk import run as run_clean_chunk
from ingestion.embed_store import run as run_embed_store
from utils.config_loader import settings


def main():
    from utils.index_audit import validate_storage_paths
    validate_storage_paths(settings.paths.chroma_dir, settings.paths.source_store_dir)
    print("=" * 90)
    print("RAG INGEST")
    print("=" * 90)

    print("\n[1/2] 多格式解析、清洗与章节切片")
    run_clean_chunk(
        kb_root=settings.paths.kb_root,
        output_path=settings.paths.chunk_jsonl,
        error_path=settings.paths.error_jsonl,
        section_max_chars=settings.ingestion.chunking.section_max_chars,
    )

    if (
        not settings.paths.chunk_jsonl.exists()
        or settings.paths.chunk_jsonl.stat().st_size == 0
    ):
        raise RuntimeError(
            "清洗/切片阶段没有生成任何 Chunk，已停止 Embedding。\n"
            f"请检查错误日志：{settings.paths.error_jsonl}"
        )

    print("\n[2/2] Embedding 与分域向量库入库")
    run_embed_store(settings.paths.chunk_jsonl)

    print("\n" + "=" * 90)
    print("RAG 入库完成")
    print("=" * 90)
    print(f"无正文索引：{settings.paths.chunk_jsonl}")
    print(f"错误日志：{settings.paths.error_jsonl}")
    print("\n接下来可以执行：")
    print('python search.py "BUCK电感如何计算？"')


if __name__ == "__main__":
    main()
