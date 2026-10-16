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

import argparse

from ingestion.clean_chunk import run as run_clean_chunk
from ingestion.embed_store import run as run_embed_store
from utils.config_loader import settings


def main(argv=None):
    parser = argparse.ArgumentParser(description="RAG 一键入库（解析→切片→Embedding→向量库）")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从中断的切片进度继续：未变的文件直接复用，增删改过的文件会被正确处理",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="增量：只重新解析新增/内容变化的文件，并删除已不存在文件的 chunk",
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
    # argv=None 时读 sys.argv（CLI 用法）；程序化调用请显式传参（如 main([])），
    # 避免把宿主进程的 sys.argv 误当成入库参数。
    args = parser.parse_args(argv)

    from utils.index_audit import validate_storage_paths
    validate_storage_paths(settings.paths.chroma_dir, settings.paths.source_store_dir)
    print("=" * 90)
    print("RAG INGEST")
    print("=" * 90)

    print("\n[1/2] 多格式解析、清洗与章节切片")
    summary = run_clean_chunk(
        kb_root=settings.paths.kb_root,
        output_path=settings.paths.chunk_jsonl,
        error_path=settings.paths.error_jsonl,
        section_max_chars=settings.ingestion.chunking.section_max_chars,
        resume=args.resume,
        incremental=args.incremental,
        dry_run=args.dry_run,
        force_delete=args.force_delete,
    )

    if args.dry_run:
        print("\n[dry-run] 未写入任何文件，已结束。")
        return

    if summary.get("mode") == "incremental":
        print(
            f"\n[1/2] 增量摘要：新增 {summary['added']}，重切 {summary['changed']}，"
            f"删除 {summary['deleted']}，未变 {summary['unchanged']}。"
        )

    if (
        not settings.paths.chunk_jsonl.exists()
        or settings.paths.chunk_jsonl.stat().st_size == 0
    ):
        raise RuntimeError(
            "清洗/切片阶段没有生成任何 Chunk，已停止 Embedding。\n"
            f"请检查错误日志：{settings.paths.error_jsonl}"
        )

    # 第二阶段无条件执行：它是对账器（比对索引期望的 chunk 集合与向量库现存内容），
    # 空转成本很低，但跳过会让「切片已成功、入库中途失败」的中间状态永不自愈。
    print("\n[2/2] Embedding 与分域向量库对账")
    run_embed_store(settings.paths.chunk_jsonl)

    print("\n" + "=" * 90)
    print("RAG 入库完成")
    print("=" * 90)
    print(f"无正文索引：{settings.paths.chunk_jsonl}")
    print(f"错误日志：{settings.paths.error_jsonl}")
    print("\n接下来可以执行：")
    print('python -m pipeline.retrieval.retriever "buck电感如何计算？"')


if __name__ == "__main__":
    main()
