# -*- coding: utf-8 -*-
"""
RAG 一键入库入口

执行：
    python ingest.py

内部自动完成：
1. 扫描知识库
2. 多格式解析
3. 清洗
4. 切片
5. 输出标准 Chunk JSONL
6. Embedding
7. 写入 Chroma 向量库

用户无需分别执行 clean_chunk.py 和 embed_store.py。
"""

from __future__ import annotations

from clean_chunk import run as run_clean_chunk
from embed_store import run as run_embed_store
from config import (
    KB_ROOT,
    CHUNK_JSONL,
    ERROR_JSONL,
    MAX_CHARS,
    OVERLAP_CHARS,
)


def main():
    print("=" * 90)
    print("RAG INGEST")
    print("=" * 90)

    print("\n[1/2] 多格式解析、清洗与切片")
    run_clean_chunk(
        kb_root=KB_ROOT,
        output_path=CHUNK_JSONL,
        error_path=ERROR_JSONL,
        max_chars=MAX_CHARS,
        overlap_chars=OVERLAP_CHARS,
    )

    print("\n[2/2] Embedding 与向量库入库")
    run_embed_store(CHUNK_JSONL)

    print("\n" + "=" * 90)
    print("RAG 入库完成")
    print("=" * 90)
    print(f"标准 Chunk：{CHUNK_JSONL}")
    print(f"错误日志：{ERROR_JSONL}")
    print("\n接下来可以执行：")
    print('python search.py "BUCK电感如何计算？"')


if __name__ == "__main__":
    main()
