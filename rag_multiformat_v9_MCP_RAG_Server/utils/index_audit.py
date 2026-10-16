"""无正文索引的校验与审计：python index_audit.py。"""
from pathlib import Path
import re

from utils.config_loader import settings


INDEX_FIELDS = frozenset("""document_id domain subdomain document_type source_file relative_path source_format
section_id section_chunk_index section_chunk_count content_type page page_end
sheet_name row_start row_end slide_number image_index media_start_sec media_end_sec
chunk_index char_count content_hash source_ref source_unit source_file_hash
line_start line_end column_start column_end char_start char_end line_basis
index_schema_version keywords
""".split())


def validate_storage_paths(chroma_dir: Path, source_dir: Path):
    chroma = Path(chroma_dir).resolve()
    source = Path(source_dir).resolve()
    if source.is_relative_to(chroma) or chroma.is_relative_to(source):
        raise ValueError("settings.paths.source_store_dir 与 settings.paths.chroma_dir 必须是互不包含的独立目录")


def assert_index_collection(collection, scan: bool = False) -> int:
    if (collection.metadata or {}).get("index_schema_version") != settings.paths.index_schema_version:
        raise RuntimeError("检测到旧版/未知 Chroma 集合。请使用全新的 RAG_CHROMA_DIR 重新入库，不能在旧库原地覆盖。")
    if not scan:
        return 0
    count = 0
    while True:
        result = collection.get(limit=256, offset=count, include=["documents", "metadatas"])
        ids = result.get("ids") or []
        if not ids:
            return count
        documents = result.get("documents")
        if documents is not None and any(document is not None for document in documents):
            raise RuntimeError("审计失败：向量库 documents 含文本，请在全新目录重建")
        metadatas = result.get("metadatas") or []
        if len(metadatas) != len(ids):
            raise RuntimeError("审计失败：索引缺少 metadata")
        for meta in metadatas:
            if not meta or set(meta) - INDEX_FIELDS:
                raise RuntimeError("审计失败：metadata 包含非白名单字段")
            ref = str(meta.get("source_ref", ""))
            if meta.get("index_schema_version") != settings.paths.index_schema_version or not re.fullmatch(r"[0-9a-f]{64}", ref):
                raise RuntimeError("审计失败：缺少新版原文定位信息")
            # 定位信息不仅要格式合法，还必须能解析到真实存在的 source_store 快照，
            # 否则即为悬空引用（正文已丢失），检索时会静默失败。
            snapshot = Path(settings.paths.source_store_dir) / ref[:2] / (ref + ".json")
            if not snapshot.is_file():
                raise RuntimeError("审计失败：source_ref 指向的原文快照缺失（悬空引用），请重新入库")
        count += len(ids)


def _rag_collections(client) -> dict:
    prefix = settings.paths.collection_prefix
    found = {}
    for item in client.list_collections():
        name = item if isinstance(item, str) else item.name
        if name.startswith(prefix):
            found[name[len(prefix):]] = name
    return found


def assert_index_matches_index_file(client, chunk_jsonl: Path | None = None) -> int:
    """比对向量库与无正文索引的 chunk_id 集合，多出或缺少都报错。

    这是唯一能发现「幽灵向量」的检查。``assert_index_collection`` 只验证
    source_ref 是否指向真实快照，而快照只增不删、永不悬空，所以「文件已从知识库
    删除、向量却还在」这种状态在那里必然 PASS。
    """
    from ingestion.index_state import scan_index_ids

    index_path = Path(chunk_jsonl if chunk_jsonl is not None else settings.paths.chunk_jsonl)
    if not index_path.exists():
        raise RuntimeError(f"找不到无正文索引：{index_path}")

    by_domain = scan_index_ids(index_path)
    expected_total = sum(len(ids) for ids in by_domain.values())
    actual_total = 0
    collections = _rag_collections(client)

    problems = []
    for domain in sorted(set(collections) | set(by_domain)):
        expected = by_domain.get(domain, set())
        name = collections.get(domain)
        if name is None:
            actual = set()
        else:
            actual = set(client.get_collection(name=name, embedding_function=None).get(include=[])["ids"])
        actual_total += len(actual)

        extra = sorted(actual - expected)
        missing = sorted(expected - actual)
        if extra or missing:
            detail = []
            if extra:
                detail.append(f"向量库多出 {len(extra)}（前几个：{', '.join(extra[:3])}）")
            if missing:
                detail.append(f"向量库缺少 {len(missing)}（前几个：{', '.join(missing[:3])}）")
            problems.append(f"[{domain}] " + "；".join(detail))

    if problems:
        raise RuntimeError(
            "审计失败：向量库与索引不一致（多出即幽灵向量，缺少即未入库）：\n  "
            + "\n  ".join(problems)
            + "\n请运行 python -m ingestion.embed_store 对账（不需要重新解析文件）。"
        )

    if actual_total != expected_total:
        raise RuntimeError(
            f"审计失败：向量库合计 {actual_total} 条，索引合计 {expected_total} 条"
        )
    return expected_total


def main():
    import chromadb
    validate_storage_paths(settings.paths.chroma_dir, settings.paths.source_store_dir)
    if not settings.paths.chroma_dir.exists():
        raise SystemExit("向量库尚不存在，请先运行 ingest.py")
    client = chromadb.PersistentClient(path=str(settings.paths.chroma_dir))
    total = 0
    collections = 0
    for name in _rag_collections(client).values():
        collection = client.get_collection(name=name, embedding_function=None)
        total += assert_index_collection(collection, scan=True)
        collections += 1
    if not collections:
        raise SystemExit("没有可审计的 RAG 集合，请先运行 ingest.py")
    records = assert_index_matches_index_file(client)
    print(f"PASS: {collections} collections, {total} records; documents=None; metadata 白名单通过。")
    print(f"PASS: 向量库与索引一致，共 {records} 条记录。")
    print("这是当前逻辑记录审计；不代表旧数据库文件、WAL、备份中的历史正文已被擦除。")


if __name__ == "__main__":
    main()
