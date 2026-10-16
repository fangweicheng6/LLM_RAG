# -*- coding: utf-8 -*-
"""RAG 内部上下文补全执行器。

优先级：
1. same_section
2. parent_section
3. targeted_search
4. neighbors

本模块不负责判断“缺什么”，只执行 Context Selector 给出的结构化补全请求。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List
from ingestion.source_store import hydrate_items
from utils.index_audit import assert_index_collection

from utils.config_loader import settings


_PRIORITY = {
    "same_section": 0,
    "parent_section": 1,
    "targeted_search": 2,
    "neighbors": 3,
}


def _collection_name_from_item(item: Dict[str, Any]) -> str:
    explicit = str(item.get("collection", "")).strip()
    if explicit:
        return explicit
    meta = item.get("metadata") or {}
    domain = str(meta.get("domain", "")).strip().lower()
    return settings.paths.collection_prefix + domain if domain else ""


def _result_items_from_get(
    result: Dict[str, Any],
    *,
    collection_name: str,
    retrieval_source: str,
) -> List[Dict[str, Any]]:
    ids = result.get("ids") or []
    metadatas = result.get("metadatas") or []

    items: List[Dict[str, Any]] = []
    for chunk_id, metadata in zip(ids, metadatas):
        items.append({
            "chunk_id": chunk_id,
            "metadata": metadata or {},
            "collection": collection_name,
            "retrieval_source": retrieval_source,
        })
    return hydrate_items(items)


def _get_document_chunks(client, anchor: Dict[str, Any]) -> List[Dict[str, Any]]:
    meta = anchor.get("metadata") or {}
    document_id = str(meta.get("document_id", "")).strip()
    collection_name = _collection_name_from_item(anchor)
    if not document_id or not collection_name:
        return []

    collection = client.get_collection(
        name=collection_name,
        embedding_function=None,
    )
    assert_index_collection(collection)
    result = collection.get(
        where={"document_id": document_id},
        include=["metadatas"],
    )
    items = _result_items_from_get(
        result,
        collection_name=collection_name,
        retrieval_source="document_lookup",
    )
    items.sort(
        key=lambda x: int((x.get("metadata") or {}).get("chunk_index", 10**9))
    )
    return items


def expand_same_section(client, anchor: Dict[str, Any]) -> List[Dict[str, Any]]:
    meta = anchor.get("metadata") or {}
    document_id = str(meta.get("document_id", "")).strip()
    section_id = str(meta.get("section_id", "")).strip()
    section_path = str(meta.get("section_path", "")).strip()
    collection_name = _collection_name_from_item(anchor)
    if not document_id or not collection_name:
        return []

    collection = client.get_collection(
        name=collection_name,
        embedding_function=None,
    )

    if section_id:
        where = {
            "$and": [
                {"document_id": document_id},
                {"section_id": section_id},
            ]
        }
    elif section_path:
        where = {
            "$and": [
                {"document_id": document_id},
                {"section_path": section_path},
            ]
        }
    else:
        return []

    assert_index_collection(collection)
    result = collection.get(
        where=where,
        include=["metadatas"],
    )
    items = _result_items_from_get(
        result,
        collection_name=collection_name,
        retrieval_source="same_section",
    )
    items.sort(
        key=lambda x: (
            int((x.get("metadata") or {}).get("section_chunk_index", 10**9)),
            int((x.get("metadata") or {}).get("chunk_index", 10**9)),
        )
    )
    return items


def _parent_paths(section_path: str) -> Iterable[str]:
    # section_path 由 Docling 路径以 " / " 拼接标题生成
    # （docling_chunker._prepare_docling_chunks）。父路径切分必须与该口径一致，
    # 否则 parent_section 补全对主入库路径恒返回空。
    parts = [part.strip() for part in section_path.split(" / ") if part.strip()]
    for end in range(len(parts) - 1, 0, -1):
        yield " / ".join(parts[:end])


def expand_parent_section(client, anchor: Dict[str, Any]) -> List[Dict[str, Any]]:
    meta = anchor.get("metadata") or {}
    section_path = str(meta.get("section_path", "")).strip()
    if not section_path or " / " not in section_path:
        return []

    document_chunks = _get_document_chunks(client, anchor)
    if not document_chunks:
        return []

    # 从直接父章节开始逐级向上找，第一个存在正文的父章节即返回。
    for parent_path in _parent_paths(section_path):
        matched = [
            item for item in document_chunks
            if str((item.get("metadata") or {}).get("section_path", "")).strip()
            == parent_path
        ]
        if matched:
            for item in matched:
                item["retrieval_source"] = "parent_section"
            matched.sort(
                key=lambda x: int(
                    (x.get("metadata") or {}).get("section_chunk_index", 10**9)
                )
            )
            return matched

    return []


def expand_neighbors(
    client,
    anchor: Dict[str, Any],
    *,
    before: int,
    after: int,
) -> List[Dict[str, Any]]:
    meta = anchor.get("metadata") or {}
    try:
        anchor_index = int(meta.get("chunk_index"))
    except Exception:
        return []

    document_chunks = _get_document_chunks(client, anchor)
    low = anchor_index - max(0, before)
    high = anchor_index + max(0, after)

    matched = []
    for item in document_chunks:
        item_meta = item.get("metadata") or {}
        try:
            idx = int(item_meta.get("chunk_index"))
        except Exception:
            continue
        if low <= idx <= high:
            item["retrieval_source"] = "neighbors"
            matched.append(item)

    matched.sort(
        key=lambda x: int((x.get("metadata") or {}).get("chunk_index", 10**9))
    )
    return matched


def expand_targeted_search(
    *,
    client,
    embedding_model,
    query: str,
    domains: List[str],
) -> List[Dict[str, Any]]:
    query = query.strip()
    if not query:
        return []

    query_embedding = embedding_model.encode(
        [settings.model.embedding.query_instruction + query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    result_items: List[Dict[str, Any]] = []
    for domain in domains:
        domain = domain.strip().lower()
        if domain not in settings.domains.allowed:
            continue

        collection_name = settings.paths.collection_prefix + domain
        try:
            collection = client.get_collection(
                name=collection_name,
                embedding_function=None,
            )
        except Exception:
            continue

        assert_index_collection(collection)
        count = collection.count()
        if count <= 0:
            continue

        result = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(settings.rag.context.expander.targeted_candidates, count),
            include=["metadatas", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for chunk_id, metadata, distance in zip(
            ids, metadatas, distances
        ):
            distance = float(distance)
            result_items.append({
                "chunk_id": chunk_id,
                "metadata": metadata or {},
                "collection": collection_name,
                "distance": distance,
                "similarity": 1.0 - distance,
                "retrieval_source": "targeted_search",
                "expansion_query": query,
            })

    result_items.sort(key=lambda x: x.get("distance", 10**9))
    return hydrate_items(result_items)


def _dedupe_by_chunk_id(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        chunk_id = str(item.get("chunk_id", "")).strip()
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        result.append(item)
    return result


def expand_context_requests(
    *,
    requests: List[Dict[str, Any]],
    known_contexts: List[Dict[str, Any]],
    routed_domains: List[str],
    client,
    embedding_model,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """按固定优先级执行补全请求，返回新增上下文与执行日志。"""
    by_id = {
        str(item.get("chunk_id", "")): item
        for item in known_contexts
        if item.get("chunk_id")
    }

    sorted_requests = sorted(
        requests,
        key=lambda req: _PRIORITY.get(str(req.get("type", "")), 99),
    )[:settings.rag.context.expander.max_requests]

    expanded: List[Dict[str, Any]] = []
    logs: List[Dict[str, Any]] = []

    for request in sorted_requests:
        request_type = str(request.get("type", "")).strip()
        anchor_id = str(request.get("anchor_chunk_id", "")).strip()
        anchor = by_id.get(anchor_id)
        items: List[Dict[str, Any]] = []

        try:
            if request_type == "same_section" and anchor:
                items = expand_same_section(client, anchor)

            elif request_type == "parent_section" and anchor:
                items = expand_parent_section(client, anchor)

            elif request_type == "targeted_search":
                domain = str(request.get("domain", "")).strip().lower()
                if domain and domain in settings.domains.allowed:
                    domains = [domain]
                else:
                    # 跨域安全网：Selector 已判断信息缺失、又未指定具体领域时，
                    # 不局限于 Router 判定的领域，而检索全部已配置领域。
                    # Router 只是召回优化器，不应成为补全阶段的单点故障。
                    # 注意：不能退化为「只搜 anchor 所在领域」——否则缺失信息且
                    # 未指定领域的跨域补全会被 anchor 挡住，安全网形同虚设。
                    domains = [d for d in settings.domains.allowed]

                items = expand_targeted_search(
                    client=client,
                    embedding_model=embedding_model,
                    query=str(request.get("query", "")),
                    domains=domains,
                )

            elif request_type == "neighbors" and anchor:
                before = int(
                    request.get("before", settings.rag.context.expander.neighbor_before)
                )
                after = int(
                    request.get("after", settings.rag.context.expander.neighbor_after)
                )
                items = expand_neighbors(
                    client,
                    anchor,
                    before=before,
                    after=after,
                )
        except Exception as exc:
            logs.append({
                "type": request_type,
                "anchor_chunk_id": anchor_id,
                "reason": request.get("reason", ""),
                "added": 0,
                "error": str(exc),
            })
            continue

        items = _dedupe_by_chunk_id(items)
        expanded.extend(items)
        logs.append({
            "type": request_type,
            "anchor_chunk_id": anchor_id,
            "query": request.get("query", ""),
            "reason": request.get("reason", ""),
            "added": len(items),
        })

    return _dedupe_by_chunk_id(expanded), logs
