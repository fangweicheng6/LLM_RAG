# -*- coding: utf-8 -*-
"""词法检索（BM25）模块的纯函数/索引契约单测。

不触碰 source_store / Chroma，直接以 mock chunk 构造 LexicalIndex，
锁定 tokenize / BM25 打分 / 按 domain 召回 / item 形状 四条契约。
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pipeline.retrieval import lexical
from pipeline.retrieval.lexical import tokenize, _BM25, LexicalIndex, reset_lexical_index
from utils.config_loader import settings


def _mock_chunk(chunk_id, domain, content, title="T", section_path="S"):
    c = SimpleNamespace(
        chunk_id=chunk_id,
        domain=domain,
        content=content,
        title=title,
        section_path=section_path,
    )
    c.chroma_metadata = lambda: {"domain": domain, "content_hash": chunk_id + "-h"}
    return c


class TokenizeTests(unittest.TestCase):
    def test_english_tech_tokens_and_chinese_bigrams(self):
        toks = tokenize("Buck 500kHz VBT1-S12-SMT 中文文档")
        self.assertIn("buck", toks)
        self.assertIn("500khz", toks)
        self.assertIn("vbt1-s12-smt", toks)
        for bigram in ("中文", "文文", "文档"):
            self.assertIn(bigram, toks)

    def test_lowercases(self):
        self.assertIn("buck", tokenize("BUCK"))
        self.assertNotIn("BUCK", tokenize("BUCK"))

    def test_symbol_only_is_empty(self):
        self.assertEqual(tokenize("  。。、— "), [])


class BM25Tests(unittest.TestCase):
    def test_prefers_matching_doc(self):
        docs = [
            ("a", tokenize("the quick brown fox 500kHz")),
            ("b", tokenize("中文文档 中文文档 电源")),
        ]
        bm = _BM25(docs)
        q = tokenize("quick 中文")
        self.assertGreater(bm.score(q, "a"), 0.0)
        self.assertGreater(bm.score(q, "b"), 0.0)
        # 「中文」在 b 中命中两次且 idf 更高，得分应高于 a
        self.assertGreater(bm.score(q, "b"), bm.score(q, "a"))

    def test_no_overlap_scores_zero(self):
        docs = [("a", tokenize("apple banana")), ("b", tokenize("cat dog"))]
        bm = _BM25(docs)
        self.assertEqual(bm.score(tokenize("zzz"), "a"), 0.0)


class LexicalIndexTests(unittest.TestCase):
    def setUp(self):
        chunks = [
            _mock_chunk("c1", "control", "buck converter 开关频率 效率 buck"),
            _mock_chunk("c2", "control", "环路补偿 相位裕度"),
            _mock_chunk("c3", "device", "buck 降压转换器 效率"),
        ]
        self.index = LexicalIndex(chunks)

    def test_search_returns_domain_mapping_and_respects_limit(self):
        ranked = self.index.search("buck 效率", ["control", "device"], limit=1)
        self.assertEqual(set(ranked), {"control", "device"})
        for ids in ranked.values():
            self.assertLessEqual(len(ids), 1)
        # control 域内只有 c1 命中 buck
        self.assertEqual(ranked["control"], ["c1"])
        # device 域内 c3 命中
        self.assertEqual(ranked["device"], ["c3"])

    def test_search_filters_zero_score_and_unknown_domain(self):
        ranked = self.index.search("环路补偿", ["control", "nonexistent"], limit=5)
        self.assertEqual(ranked["control"], ["c2"])
        self.assertEqual(ranked["nonexistent"], [])

    def test_to_item_shape_matches_dense_candidate(self):
        item = self.index.to_item("c1", "rag_control")
        self.assertEqual(item["chunk_id"], "c1")
        self.assertEqual(item["document"], "buck converter 开关频率 效率 buck")
        self.assertEqual(item["metadata"]["title"], "T")
        self.assertEqual(item["metadata"]["section_path"], "S")
        self.assertEqual(item["metadata"]["domain"], "control")
        self.assertEqual(item["distance"], 2.0)
        self.assertEqual(item["similarity"], 0.0)
        self.assertEqual(item["retrieval_source"], "lexical")
        self.assertEqual(item["collection"], "rag_control")


class LexicalCacheTests(unittest.TestCase):
    """词法索引的进程内缓存必须随索引文件变化而失效。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.index_path = Path(self.temp.name) / "index.jsonl"
        self.index_path.write_text("first\n", encoding="utf-8")

        patcher = patch.object(settings.paths, "chunk_jsonl", self.index_path)
        patcher.start()
        self.addCleanup(patcher.stop)

        reset_lexical_index()
        self.addCleanup(reset_lexical_index)

    def test_lexical_index_rebuilds_only_when_index_changes(self):
        versions = [
            [_mock_chunk("c1", "topology", "BUCK 电感需要根据压差和频率计算")],
            [_mock_chunk("c2", "topology", "环路补偿要保证相位裕度")],
            [_mock_chunk("c3", "topology", "环路补偿要保证相位裕度")],
        ]
        loaded = []

        def fake_load(path):
            loaded.append(path)
            return versions[len(loaded) - 1]

        with patch.object(lexical, "load_chunks", side_effect=fake_load):
            first = lexical.get_lexical_index()
            self.assertIs(lexical.get_lexical_index(), first)  # 未变时复用
            self.assertEqual(len(loaded), 1)

            # 重新入库后（mtime/size 变化）必须换到新语料，否则长驻服务永远检索旧内容。
            self.index_path.write_text("second-version\n", encoding="utf-8")
            second = lexical.get_lexical_index()
            self.assertIsNot(second, first)
            self.assertEqual(len(loaded), 2)
            self.assertTrue(second.search("环路补偿", ["topology"], 5))

            self.assertIs(lexical.get_lexical_index(), second)
            self.assertEqual(len(loaded), 2)


if __name__ == "__main__":
    unittest.main()
