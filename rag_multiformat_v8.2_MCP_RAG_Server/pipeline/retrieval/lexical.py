# -*- coding: utf-8 -*-
"""词法（BM25）检索索引，与稠密向量做混合召回。

方案 A：索引不落盘，进程内按需从 source_store 水合正文、现场建倒排，
进程退出即消失——不破坏「索引不存正文、正文只在 source_store」的安全原则。
代价：进程首次查询多一次约数秒的冷启动（之后全局缓存复用）。
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List

from utils.config_loader import settings
from ingestion.embed_store import load_chunks

# 英文技术词（含型号/参数，如 VBT1-S12-S12-SMT、500kHz）+ 中文二字 bigram，
# 与 ingestion.source_store.extract_keywords 的确定性分词口径一致。
_EN_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+.\-]{0,31}")
_CJK = re.compile(r"[一-鿿]+")


def tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = _EN_TOKEN.findall(text)
    for phrase in _CJK.findall(text):
        tokens.extend(phrase[i:i + 2] for i in range(len(phrase) - 1))
    return tokens


class _BM25:
    """标准 Okapi BM25，纯词频统计，不依赖任何模型。"""

    def __init__(self, docs: List[tuple[str, List[str]]]):
        self.doc_ids = [doc_id for doc_id, _ in docs]
        self.doc_len: Dict[str, int] = {}
        self.doc_tf: Dict[str, Dict[str, int]] = {}
        self.doc_freq: Dict[str, int] = defaultdict(int)
        total = 0
        for doc_id, tokens in docs:
            tf = Counter(tokens)
            self.doc_len[doc_id] = len(tokens)
            self.doc_tf[doc_id] = dict(tf)
            total += len(tokens)
            for term in tf:
                self.doc_freq[term] += 1
        self.n = len(self.doc_ids)
        self.avgdl = total / self.n if self.n else 0.0

    def score(self, query_tokens: List[str], doc_id: str,
              k1: float = 1.5, b: float = 0.75) -> float:
        dl = self.doc_len.get(doc_id, 0)
        if dl == 0:
            return 0.0
        tf_map = self.doc_tf.get(doc_id, {})
        score = 0.0
        for term in query_tokens:
            df = self.doc_freq.get(term, 0)
            if df == 0:
                continue
            tf = tf_map.get(term, 0)
            if tf == 0:
                continue
            idf = math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))
            score += idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / self.avgdl))
        return score


class LexicalIndex:
    """进程内 BM25 索引：chunk_id -> 词频统计，支持按 domain 召回。"""

    def __init__(self, chunks):
        self._chunks = {c.chunk_id: c for c in chunks}
        self._domain_ids: Dict[str, List[str]] = defaultdict(list)
        docs = []
        for chunk in chunks:
            self._domain_ids[chunk.domain].append(chunk.chunk_id)
            docs.append((chunk.chunk_id, tokenize(chunk.content)))
        self._bm25 = _BM25(docs)

    def search(self, query: str, domains: List[str], limit: int) -> Dict[str, List[str]]:
        """返回 {domain: [chunk_id, ...]}，每个 domain 取词法分最高的 limit 个。"""
        query_tokens = tokenize(query)
        if not query_tokens:
            return {}
        out: Dict[str, List[str]] = {}
        for domain in domains:
            ids = self._domain_ids.get(domain, [])
            scored = [(self._bm25.score(query_tokens, cid), cid) for cid in ids]
            scored = [(s, cid) for s, cid in scored if s > 0.0]
            scored.sort(key=lambda x: (-x[0], x[1]))
            out[domain] = [cid for _, cid in scored[:limit]]
        return out

    def to_item(self, chunk_id: str, collection: str) -> dict:
        """转成与稠密候选（query_collection 输出）同构的 item。"""
        chunk = self._chunks[chunk_id]
        meta = chunk.chroma_metadata()
        meta["title"] = chunk.title
        meta["section_path"] = chunk.section_path
        return {
            "chunk_id": chunk_id,
            "metadata": meta,
            "document": chunk.content,
            # 伪距离取 2.0（余弦距离上界）：同一 chunk 若被稠密与词法同时命中，
            # deduplicate_candidate_pool 会保留稠密条目，避免丢失真实向量相似度。
            # 代价：关掉 rerank 时词法条目恒排在所有稠密条目之后，可能被 selector
            # 的字符预算截断——这是「稠密优先」的明确取舍；默认 rerank 开启时会
            # 用交叉编码器重新打分，词法条目不会因此被埋没。
            "distance": 2.0,
            "similarity": 0.0,
            "retrieval_source": "lexical",
            "collection": collection,
        }


_index: LexicalIndex | None = None


def get_lexical_index() -> LexicalIndex:
    """惰性构建并全局缓存词法索引（方案 A：进程内、不落盘）。"""
    global _index
    if _index is None:
        chunks = load_chunks(settings.paths.chunk_jsonl)
        _index = LexicalIndex(chunks)
    return _index


def lexical_search(query: str, domains: List[str], limit: int) -> List[dict]:
    """对给定 domain 做词法召回，返回与稠密候选同构的 item 列表。"""
    if not query.strip() or limit <= 0:
        return []
    index = get_lexical_index()
    ranked = index.search(query, domains, limit)
    items: List[dict] = []
    for domain, chunk_ids in ranked.items():
        collection = settings.paths.collection_prefix + domain
        for chunk_id in chunk_ids:
            items.append(index.to_item(chunk_id, collection))
    return items
