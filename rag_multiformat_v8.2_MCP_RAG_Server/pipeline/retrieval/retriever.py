# -*- coding: utf-8 -*-
"""
阶段 3：LLM Router + 分域向量召回 + LLM Context Selector + Answer Generator

流程：
用户问题
-> LLM 提取关键词（仅用于查询分析/日志，暂不参与首轮向量检索）
-> LLM Domain Router 只根据原始问题判断 topology / device / pcb / control / schematic 一个或多个领域
-> 仅使用原始问题做首轮 Query Embedding
-> 进入对应 Chroma Collection 生成候选结果池
-> LLM 从候选结果池按“最小充分”原则选择相关上下文并判断完整性
-> 若不完整：same_section / parent_section / targeted_search / neighbors 自适应补全
-> LLM 对补全结果再次选择
-> LLM 基于最终上下文生成回答

不再使用：
- Embedding Router threshold
- CrossEncoder Reranker
- 最终固定 Top-K Context
"""

from __future__ import annotations

import argparse
import contextlib
import io
import unicodedata
from typing import Any, Dict, List
from pathlib import Path


from utils.config_loader import settings
from pipeline.context.selector import select_contexts
from pipeline.context.expander import expand_context_requests
from pipeline.generation.generator import generate_answer
from llm import create_llm_client
from pipeline.routing.router import analyze_query
from ingestion.source_store import hydrate_items
from utils.index_audit import assert_index_collection
from utils.config_loader import settings
from pipeline.rag_modules import route_query, choose_contexts
from pipeline.retrieval.reranker import rerank_candidates
from pipeline.retrieval.multi_hop import expand_query_to_subqueries
from pipeline.retrieval.lexical import lexical_search
from observability import langfuse_span, update_span


# ============================================================
# Embedding / Chroma
# ============================================================

_embedding_model = None

def load_embedding_model():
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    import warnings
    # sentence-transformers / transformers 内部仍调用已废弃的 torch.jit.script，
    # 属第三方无害噪音，这里定向屏蔽，避免每次启动刷屏。
    warnings.filterwarnings(
        "ignore",
        message=r"`torch\.jit\.script` is deprecated",
        category=FutureWarning,
    )

    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Embedding model: {settings.model.embedding.name}")
    print(f"Embedding device: {device}")

    # 优先离线加载本地缓存，避免服务启动/首次请求时访问 HuggingFace
    # 网络异常导致长时间阻塞，Windows 下 Ctrl+C 也可能无法及时响应。
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


def get_client():
    import chromadb
    return chromadb.PersistentClient(
        path=str(settings.paths.chroma_dir)
    )


def get_available_domains(client) -> List[str]:
    """返回当前 Chroma 中实际存在的一级知识域。

    Router 负责语义分类；这里仅做运行时可用性过滤，
    避免 Router 选中了尚未建库的 domain 后反复打印 Collection not found。
    """
    try:
        raw_collections = client.list_collections()
    except Exception:
        return []

    collection_names = set()
    for item in raw_collections:
        if isinstance(item, str):
            name = item
        else:
            name = getattr(item, "name", "")
        name = str(name).strip()
        if name:
            collection_names.add(name)

    # Chroma 中保存的是 collection 名称（例如 rag_pcb），
    # 运行时内部使用的是 domain 名称（例如 pcb）。
    # 兼容旧版本索引：不要依赖固定 settings.domains.allowed 顺序，
    # 直接根据实际存在的 collection 去除 prefix。
    available = []
    for name in sorted(collection_names):
        if name.startswith(settings.paths.collection_prefix):
            domain = name[len(settings.paths.collection_prefix):].strip()
            if domain:
                available.append(domain)

    # 保留配置中的合法 domain 过滤，避免误读取其他 collection。
    return [
        domain for domain in available
        if domain in settings.domains.allowed
    ]


def query_collection(
    *,
    collection,
    query_embedding,
    candidate_limit: int,
) -> List[Dict[str, Any]]:
    assert_index_collection(collection)
    count = collection.count()
    if count <= 0:
        return []

    n_results = min(candidate_limit, count)

    result = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
        include=[
            "metadatas",
            "distances",
        ],
    )

    ids = result.get("ids", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    items: List[Dict[str, Any]] = []

    for chunk_id, meta, distance in zip(
        ids,
        metadatas,
        distances,
    ):
        distance = float(distance)

        items.append(
            {
                "chunk_id": chunk_id,
                "metadata": meta or {},
                "distance": distance,
                "similarity": 1.0 - distance,
                "retrieval_source": "vector",
            }
        )

    return hydrate_items(items)


def _content_key(item: Dict[str, Any]) -> str:
    """按正文内容生成去重键：优先 content_hash，否则归一化正文。"""
    meta = item.get("metadata") or {}
    content_hash = str(meta.get("content_hash", "")).strip()
    if content_hash:
        return "hash:" + content_hash
    return "text:" + " ".join(str(item.get("document", "")).lower().split())


def deduplicate_candidate_pool(
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """去除同一 chunk_id / 完全相同正文，不做语义 Rerank。"""

    def distance_value(item: Dict[str, Any]) -> float:
        value = item.get("distance")
        return float(value) if isinstance(value, (int, float)) else 1e9

    # 第一遍：同一 chunk_id（稠密与词法同时命中同一块）保留距离更小者。
    best: Dict[str, Dict[str, Any]] = {}
    for item in candidates:
        chunk_id = str(item.get("chunk_id", "")).strip()
        key = "id:" + chunk_id if chunk_id else _content_key(item)
        previous = best.get(key)
        if previous is None or distance_value(item) < distance_value(previous):
            best[key] = item

    # 第二遍：正文完全相同但 chunk_id 不同（重复文档/重复内容）保留距离更小者。
    # 早期实现只按 chunk_id 去重，而 chunk_id 恒非空，导致「同正文去重」分支
    # 永远不触发；这里补上真正的正文级去重。
    best_by_content: Dict[str, Dict[str, Any]] = {}
    for item in best.values():
        key = _content_key(item)
        previous = best_by_content.get(key)
        if previous is None or distance_value(item) < distance_value(previous):
            best_by_content[key] = item

    result = list(best_by_content.values())
    result.sort(key=distance_value)
    return result


# ============================================================
# 终端显示
# ============================================================

def display_width(text: str) -> int:
    width = 0
    for ch in str(text):
        if unicodedata.east_asian_width(ch) in ("W", "F", "A"):
            width += 2
        else:
            width += 1
    return width


def pad(text, width: int) -> str:
    text = "" if text is None else str(text)
    current_width = display_width(text)

    if current_width >= width:
        return text

    return text + " " * (width - current_width)


def print_contexts(
    query: str,
    analysis: Dict[str, Any],
    selected_contexts: List[Dict[str, Any]],
    selector_reason: str,
) -> None:
    print("\n" + "=" * 100)
    print(f"Query: {query}")
    print(f"Keywords: {', '.join(analysis['keywords'])}")
    router_domains = analysis.get("router_domains") or analysis["domains"]
    if router_domains != analysis["domains"]:
        print(f"Router domains: {', '.join(router_domains)}")
    print(f"Domains: {', '.join(analysis['domains'])}")

    if analysis.get("reason"):
        print(f"Router reason: {analysis['reason']}")

    if selector_reason:
        print(f"Context selector: {selector_reason}")

    print("=" * 100)

    if not selected_contexts:
        print("LLM 没有从候选结果池中选择任何相关上下文。")
        return

    left_width = 42

    for index, item in enumerate(selected_contexts, start=1):
        meta = item.get("metadata") or {}

        similarity = item.get("similarity")
        similarity_text = (
            f"{float(similarity):.4f}"
            if isinstance(similarity, (int, float))
            else "n/a"
        )
        print(
            f"\n[Selected {index}] "
            f"chunk_id={item.get('chunk_id', '')} "
            f"vector_similarity={similarity_text} "
            f"retrieval_source={item.get('retrieval_source', 'vector')}"
        )

        left = f"collection={item.get('collection', '')}"
        right = f"domain={meta.get('domain', '')}"
        print(f"{pad(left, left_width)} | {right}")

        left = f"subdomain={meta.get('subdomain', '')}"
        right = f"document_type={meta.get('document_type', '')}"
        print(f"{pad(left, left_width)} | {right}")

        left = f"source={meta.get('source_file', '')}"
        right = f"content_type={meta.get('content_type', '')}"
        print(f"{pad(left, left_width)} | {right}")

        left = "section"
        right = meta.get("section_path", "")
        print(f"{pad(left, left_width)} | {right}")

        if meta.get("section_chunk_count") is not None:
            section_index = int(meta.get("section_chunk_index", 0)) + 1
            section_count = int(meta.get("section_chunk_count", 1))
            print(
                f"{pad('section_part', left_width)} | "
                f"{section_index}/{section_count}"
            )

        if meta.get("page") is not None:
            page = meta.get("page")
            page_end = meta.get("page_end")
            right = (
                f"{page}-{page_end}"
                if page_end is not None and page_end != page
                else str(page)
            )
            print(f"{pad('page', left_width)} | {right}")

        print(f"{pad('relative_path', left_width)} | {meta.get('relative_path', '')}")
        print(f"{pad('parsed_text_lines', left_width)} | "
              f"unit={meta.get('source_unit')} lines={meta.get('line_start')}-{meta.get('line_end')} "
              f"columns={meta.get('column_start')}-{meta.get('column_end')}")
        print(f"{pad('keywords', left_width)} | {meta.get('keywords', '')}")

        if meta.get("sheet_name"):
            left = f"sheet={meta.get('sheet_name')}"
            right = f"rows={meta.get('row_start')}-{meta.get('row_end')}"
            print(f"{pad(left, left_width)} | {right}")

        if meta.get("slide_number") is not None:
            print(
                f"{pad('slide', left_width)} | "
                f"{meta.get('slide_number')}"
            )

        if meta.get("media_start_sec") is not None:
            print(
                f"{pad('time', left_width)} | "
                f"{meta.get('media_start_sec')}-{meta.get('media_end_sec')}s"
            )

        print("-" * 100)
        print(item.get("document", ""))


def print_final_answer(
    answer: str,
    sources: List[Dict[str, str]],
) -> None:
    print("\n" + "=" * 100)
    print("Final Answer")
    print("=" * 100)
    print(answer)

    if sources:
        print("\nSources")
        for source in sources:
            print(
                f"[{source['label']}] "
                f"{source['source_file']} | "
                f"{source['section']} | "
                f"{source.get('location', '')} | "
                f"chunk_id={source['chunk_id']}"
            )


# ============================================================
# 主检索流程
# ============================================================

def _run(query: str, *, llm=None, client=None, model=None, allow_empty=False, strict=False, stop_after_retrieval=False):
    query = query.strip()
    if not query:
        raise ValueError("Query 不能为空。")

    if llm is None:
        # 按角色分别建客户端：router / multi_hop 用轻量模型，selector / generator 用强推理模型。
        # 各角色模型名统一在 configs/model.yaml 的 model.llm.roles 配置，其余参数共享。
        router_llm = create_llm_client(role="router")
        multi_hop_llm = create_llm_client(role="multi_hop")
        selector_llm = create_llm_client(role="selector")
        generator_llm = create_llm_client(role="generator")
    else:
        # 显式传入 llm（如测试注入 mock、评估传入指定客户端）时，各角色共用同一客户端，保持向后兼容。
        router_llm = multi_hop_llm = selector_llm = generator_llm = llm
    client = client if client is not None else get_client()
    available_domains = get_available_domains(client)
    if not available_domains:
        raise RuntimeError("没有可用的知识域 Collection，请先运行 python ingest.py。")

    # 1) Query Analysis：关键词提取与 Domain Router 并行解耦。
    #    - keywords：仅用于日志/可解释性，并为未来 BM25/混合检索预留；
    #    - domains：Domain Router 只根据原始 Query 判断，不读取 keywords。
    with langfuse_span("router", input={"query": query}) as router_span:
        analysis = route_query(query, router_llm, available_domains, analyzer=analyze_query)
        update_span(router_span, {"domains": analysis.get("domains", []), "keywords": analysis.get("keywords", [])})

    router_domains = list(analysis["domains"])
    # 无需再校验 domain 是否合法：classify_domains 已过滤到 settings.domains.allowed，
    # route_query 在 Router 关闭时也只返回 available_domains（⊆ allowed）。
    # 这里保留原始 Router 结果用于调试，实际检索领域由下方可用性过滤决定。

    # Router 做“语义上应该去哪”，Chroma 可用性过滤做“当前实际上能去哪”。
    # 二者职责分离：不会为了迁就当前建库状态而让 Router 把 device 问题误分到 topology。
    retrieval_domains = [
        domain for domain in router_domains
        if domain in available_domains
    ]
    skipped_domains = [
        domain for domain in router_domains
        if domain not in retrieval_domains
    ]

    # Router 是检索优化器，不作为系统单点故障。
    # 当 Router 无法判断领域，或选中的领域无可用 Collection 时，
    # 自动退化为全部可用领域检索。
    if not retrieval_domains:
        retrieval_domains = list(available_domains)
        analysis["domains"] = retrieval_domains
        analysis["reason"] = (
            str(analysis.get("reason", "")).strip()
            + "；触发全局领域检索 fallback。"
        ).strip("；")

    # 后续检索/补全只使用当前实际可用的领域；同时保留 Router 原始结果用于调试。
    analysis["router_domains"] = router_domains
    analysis["domains"] = retrieval_domains

    print("\nLLM Query Analysis")
    print(f"  keywords        : {', '.join(analysis['keywords']) or '无'}")
    print(f"  router domains  : {', '.join(router_domains)}")
    print(f"  retrieval domains: {', '.join(retrieval_domains)}")
    print(f"  vector query     : {query} (original query only)")
    if skipped_domains:
        print(
            "  unavailable     : "
            + ", ".join(skipped_domains)
            + " (对应 Collection 尚不存在，已跳过)"
        )
    if analysis.get("reason"):
        print(f"  reason          : {analysis['reason']}")

    # 2) 首轮 Query Embedding：只使用用户原始 Query。
    #    不再拼接 keywords，避免同义词重复加权或关键词扩写改变原始意图。
    model = model if model is not None else load_embedding_model()
    retrieval_queries = expand_query_to_subqueries(query, multi_hop_llm, analysis["domains"])
    query_embeddings = model.encode(
        [settings.model.embedding.query_instruction + q for q in retrieval_queries],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # 3) 生成第一轮候选池。
    #    cross_domain_enabled 时检索全部可用领域：Router 对跨域问题容易漏判
    #    （例如把 pcb+topology 误判为 schematic），只检索 Router 领域会丢掉跨域金块。
    #    Router 仍用于生成领域视角子问题；跨域时全域检索，交给 rerank+selector 精排。
    if settings.rag.retrieval.multi_hop.cross_domain_enabled:
        search_domains = list(available_domains)
    else:
        search_domains = list(retrieval_domains)

    with langfuse_span(
        "retrieval",
        input={
            "domains": search_domains,
            "query_count": len(retrieval_queries),
        },
    ) as retrieval_span:
        candidate_pool: List[Dict[str, Any]] = []

        for domain in search_domains:
            collection_name = settings.paths.collection_prefix + domain
            try:
                collection = client.get_collection(
                    name=collection_name,
                    embedding_function=None,
                )
            except Exception as e:
                if strict:
                    raise RuntimeError(f"无法加载 collection={collection_name}") from e
                print(f"[WARN] 无法加载 collection={collection_name}: {e}")
                continue

            domain_items = []
            for query_embedding in query_embeddings:
                items = query_collection(
                    collection=collection,
                    query_embedding=query_embedding,
                    candidate_limit=settings.rag.retrieval.candidate_pool_size_per_domain,
                )
                domain_items.extend(items)
            for item in domain_items:
                item["collection"] = collection_name
            candidate_pool.extend(domain_items)

        # 词法（BM25）混合召回：与稠密候选并集后统一交给 reranker 重排。
        # 词法索引进程内按需构建（方案 A，不落盘），失败时退回纯稠密检索。
        if settings.rag.retrieval.hybrid.enabled:
            try:
                candidate_pool.extend(
                    lexical_search(
                        query,
                        search_domains,
                        settings.rag.retrieval.hybrid.lexical_candidates,
                    )
                )
            except Exception as exc:
                print(f"[WARN] 词法召回不可用，退回纯稠密检索: {exc}")

        if not candidate_pool:
            # 检索层容错：领域检索没有召回时，不直接终止。
            # Router 只是缩小范围，不能成为单点故障。
            # 尝试一次全域检索。
            if retrieval_domains != available_domains:
                print("[WARN] 指定领域无召回，触发全局检索 fallback")
                fallback_candidates: List[Dict[str, Any]] = []
                for domain in available_domains:
                    collection_name = settings.paths.collection_prefix + domain
                    try:
                        collection = client.get_collection(
                            name=collection_name,
                            embedding_function=None,
                        )
                    except Exception:
                        continue
                    for query_embedding in query_embeddings:
                        items = query_collection(
                            collection=collection,
                            query_embedding=query_embedding,
                            candidate_limit=settings.rag.retrieval.candidate_pool_size_per_domain,
                        )
                        for item in items:
                            item["collection"] = collection_name
                        fallback_candidates.extend(items)
                candidate_pool.extend(fallback_candidates)

            if not candidate_pool:
                if allow_empty:
                    return _empty_result(query, analysis, router_domains, retrieval_domains,
                                         "向量检索未召回候选块")
                # 最终失败才抛异常。
                raise RuntimeError(
                    "全局 fallback 后仍没有召回候选结果，请确认索引状态。"
                )

        before_dedup = len(candidate_pool)
        candidate_pool = deduplicate_candidate_pool(candidate_pool)
        candidate_pool = rerank_candidates(query, candidate_pool)
        update_span(
            retrieval_span,
            {
                "candidate_count_before_dedup": before_dedup,
                "candidate_count_after_rerank": len(candidate_pool),
                "collections": sorted(list({x.get("collection", "") for x in candidate_pool})),
                "top_chunks": [
                    {
                        "chunk_id": str(x.get("chunk_id", "")),
                        "collection": x.get("collection", ""),
                        "score": x.get("rerank_score", x.get("similarity", None)),
                    }
                    for x in candidate_pool[:5]
                ],
            },
        )
    removed_duplicates = before_dedup - len(candidate_pool)
    print(
        f"Candidate pool: {len(candidate_pool)} chunks"
        + (
            f" (removed {removed_duplicates} exact duplicates)"
            if removed_duplicates else ""
        )
    )

    # 专项 Retrieval benchmark 在这里截断：
    # 完整复用 Router -> multi-hop -> embedding -> Chroma -> dedup -> reranker，
    # 但不调用 Context Selector 和 Answer Generator，避免检索专项被生成阶段污染。
    if stop_after_retrieval:
        return {
            "router_enabled": settings.rag.router.enabled,
            "query": query,
            "keywords": analysis["keywords"],
            "router_domains": router_domains,
            "domains": analysis["domains"],
            "candidate_pool": candidate_pool,
            "selected_contexts": [],
            "answer": "",
            "answer_sources": [],
        }

    # 4) 第一次 Context Selector：选择 + 判断是否完整 + 生成补全请求。
    with langfuse_span("context_selector", input={"candidate_count": len(candidate_pool)}) as selector_span:
        selection, selected_contexts = choose_contexts(
            selector=select_contexts,
            query=query,
            candidates=candidate_pool,
            llm=selector_llm,
            allow_expansion=settings.rag.context.expander.max_rounds > 0,
        )

        expansion_history: List[Dict[str, Any]] = []
        expansion_round = 0

        # 5) 自适应上下文补全。第一版默认最多 1 轮。
        while (
            not selection.get("sufficient", False)
            and selection.get("expansion_requests")
            and expansion_round < settings.rag.context.expander.max_rounds
        ):
            expansion_round += 1
            print(f"\nContext expansion round {expansion_round}")

            expanded_contexts, expansion_logs = expand_context_requests(
                requests=selection.get("expansion_requests", []),
                known_contexts=candidate_pool + selected_contexts,
                routed_domains=analysis["domains"],
                client=client,
                embedding_model=model,
            )
            expansion_history.extend(expansion_logs)
            if strict and any(log.get("error") for log in expansion_logs):
                raise RuntimeError("上下文补全失败，参见 expansion 日志；此次评估样本不计作有效得分。")

            for log in expansion_logs:
                detail = log.get("query") or log.get("anchor_chunk_id") or ""
                suffix = f" error={log['error']}" if log.get("error") else ""
                print(
                    f"  {log.get('type', '')}: {detail} "
                    f"-> +{log.get('added', 0)} chunks{suffix}"
                )

            if not expanded_contexts:
                print("  没有取得新的补全上下文，停止继续补全。")
                break

            # 补全后不重新把整个初始候选池都塞给 LLM，
            # 只使用“第一次已选择内容 + 定向补全结果”做第二次精确判断。
            reselection_pool = deduplicate_candidate_pool(
                selected_contexts + expanded_contexts
            )

            selection, selected_contexts = choose_contexts(
                selector=select_contexts,
                query=query,
                candidates=reselection_pool,
                llm=selector_llm,
                allow_expansion=(
                    expansion_round < settings.rag.context.expander.max_rounds
                ),
            )

        update_span(selector_span, {"selected_chunks": len(selected_contexts), "sufficient": selection.get("sufficient", False)})
    context_sufficient = bool(selection.get("sufficient", False))
    missing_information = selection.get("missing_information", []) or []

    print(f"\nContext sufficient: {'yes' if context_sufficient else 'no'}")
    if missing_information:
        print("Missing information: " + "; ".join(missing_information))

    print_contexts(
        query=query,
        analysis=analysis,
        selected_contexts=selected_contexts,
        selector_reason=selection.get("reason", ""),
    )

    # 6) 最终回答。
    answer, answer_sources = generate_answer(
        query=query,
        selected_contexts=selected_contexts,
        llm=generator_llm,
        context_sufficient=context_sufficient,
        missing_information=missing_information,
    )

    print_final_answer(answer=answer, sources=answer_sources)

    return {
        "router_enabled": settings.rag.router.enabled,
        "llm_chunk_selection_enabled": settings.rag.context.selector.enabled,
        "query": query,
        "keywords": analysis["keywords"],
        "router_domains": router_domains,
        "domains": analysis["domains"],
        "candidate_pool": candidate_pool,
        "selected_contexts": selected_contexts,
        "selection_reason": selection.get("reason", ""),
        "context_sufficient": context_sufficient,
        "missing_information": missing_information,
        "expansion_history": expansion_history,
        "answer": answer,
        "answer_sources": answer_sources,
    }


def _empty_result(query, analysis, router_domains, domains, reason):
    """有效的检索未命中保留路由轨迹；基础设施异常依然抛出。"""
    return {
        "query": query, "keywords": analysis.get("keywords", []),
        "router_domains": router_domains, "domains": domains,
        "candidate_pool": [], "selected_contexts": [],
        "selection_reason": reason, "context_sufficient": False,
        "missing_information": [], "expansion_history": [],
        "answer": "当前知识库候选结果中没有足够的相关上下文，无法基于知识库可靠回答该问题。",
        "answer_sources": [], "retrieval_miss_reason": reason,
        "router_enabled": settings.rag.router.enabled,
        "llm_chunk_selection_enabled": settings.rag.context.selector.enabled,
    }


def run(query: str, *, llm=None, client=None, model=None, quiet=False,
        allow_empty=False, strict=False):
    """原 run(query) 继续可用；评估入口可复用模型/客户端并静默运行。"""
    with contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext():
        return _run(query, llm=llm, client=client, model=model,
                    allow_empty=allow_empty, strict=strict)


def run_retrieval(query: str, *, llm=None, client=None, model=None, quiet=False,
                  allow_empty=True, strict=True):
    """只运行到 reranker 后的首轮候选池，用于独立 Retrieval benchmark。"""
    with contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext():
        return _run(query, llm=llm, client=client, model=model,
                    allow_empty=allow_empty, strict=strict, stop_after_retrieval=True)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="LLM Router + 分域 RAG + 自适应 Context Expansion + Answer Generator"
    )
    parser.add_argument(
        "query",
        type=str,
    )

    args = parser.parse_args()
    run(args.query)


if __name__ == "__main__":
    main()
