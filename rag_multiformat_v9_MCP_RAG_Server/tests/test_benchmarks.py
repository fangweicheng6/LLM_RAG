import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from utils.config_loader import settings
from evaluation.benchmark_evaluators import _ranking_scores, evaluate_generation, evaluate_router
from evaluation.benchmark_splitter import split_benchmarks


class FakeRouterLLM:
    def chat_json(self, messages, **kwargs):
        return {"domains": ["pcb"], "reason": "test"}


class FakeAnswerLLM:
    def chat(self, messages, **kwargs):
        return "Gold context grounded answer."


class FakeGenerationScorer:
    def score(self, row, response, gold_contexts):
        return {
            "answer_correctness": 0.8,
            "faithfulness": 0.9,
            "answer_relevancy": 0.85,
        }, {}


def full_row():
    return {
        "id": "q1",
        "user_input": "PCB 问题？",
        "reference": "标准答案",
        "expected_domains": ["pcb"],
        "reference_context_ids": ["c1"],
        "reference_contexts": ["标准上下文"],
        "reference_sources": [{"chunk_id": "c1", "domain": "pcb", "source_file": "a.md"}],
        "question_type": "single",
        "question_intent": "knowledge_qa",
        "difficulty": "easy",
        "reviewed": False,
    }


class BenchmarkTests(unittest.TestCase):
    def test_feature_flags_default_on_once(self):
        # v9 默认（对接形态，见 docs/MCP对接文档.md）：Router 关闭（全域检索），
        # Chunk Selector 开启，最终生成关闭（本服务只供 chunk，由外部系统生成回答）。
        self.assertFalse(settings.rag.router.enabled)
        self.assertTrue(settings.rag.context.selector.enabled)
        self.assertFalse(settings.rag.generation.enabled)
        # 关闭态必须走全域检索降级路径，且不调用路由 LLM。
        from pipeline.rag_modules import route_query
        analyzer = Mock()
        analysis = route_query("q", None, ["pcb", "topology"], analyzer=analyzer)
        self.assertEqual(analysis["domains"], ["pcb", "topology"])
        self.assertEqual(
            analysis["reason"],
            "Router 已关闭：检索全部现有知识域；不调用关键词提取或路由 LLM。",
        )
        analyzer.assert_not_called()

    def test_ranking_scores(self):
        s = _ranking_scores(["x", "c2", "c1"], ["c1", "c2"], (1, 2, 3))
        self.assertEqual(s["mrr"], 0.5)
        self.assertEqual(s["recall@1"], 0.0)
        self.assertEqual(s["recall@2"], 0.5)
        self.assertEqual(s["recall@3"], 1.0)
        self.assertEqual(s["hit_rate@2"], 1.0)

    def test_run_retrieval_stops_before_selector_and_answer(self):
        from pipeline.retrieval import retriever as search
        client = Mock()
        client.list_collections.return_value = ["rag_pcb"]
        client.get_collection.return_value = Mock()
        model = Mock()
        model.encode.return_value = [[1.0, 0.0]]
        llm = object()
        candidate = {
            "chunk_id": "c1", "document": "ctx", "metadata": {"domain": "pcb"},
            "distance": 0.1, "similarity": 0.9, "retrieval_source": "vector"
        }
        with patch.object(search, "analyze_query", return_value={"keywords": [], "domains": ["pcb"], "reason": "test"}), \
             patch.object(search, "expand_query_to_subqueries", return_value=["q"]), \
             patch.object(search, "query_collection", return_value=[candidate]), \
             patch.object(search, "lexical_search", return_value=[]), \
             patch.object(search, "rerank_candidates", side_effect=lambda q, x: x), \
             patch.object(search, "select_contexts", side_effect=AssertionError("selector must not run")), \
             patch.object(search, "generate_answer", side_effect=AssertionError("answer must not run")):
            result = search.run_retrieval("q", llm=llm, client=client, model=model, quiet=True)
        self.assertEqual([x["chunk_id"] for x in result["candidate_pool"]], ["c1"])
        self.assertEqual(result["answer"], "")

    def test_generation_disabled_returns_chunks_without_answer_llm(self):
        # rag.generation.enabled=false：不调用回答生成，answer 为显式 null，
        # 相关 chunk 仍经 selected_contexts 原样返回（对接外部 LLM 的路径）。
        from pipeline.retrieval import retriever as search
        client = Mock()
        client.list_collections.return_value = ["rag_pcb"]
        client.get_collection.return_value = Mock()
        model = Mock()
        model.encode.return_value = [[1.0, 0.0]]
        llm = object()
        candidate = {
            "chunk_id": "c1", "document": "ctx", "metadata": {"domain": "pcb"},
            "distance": 0.1, "similarity": 0.9, "retrieval_source": "vector"
        }
        selection = {
            "selected_chunk_ids": ["c1"], "sufficient": True,
            "missing_information": [], "expansion_requests": [], "reason": "test",
        }
        with patch.object(settings.rag.generation, "enabled", False), \
             patch.object(search, "analyze_query", return_value={"keywords": [], "domains": ["pcb"], "reason": "test"}), \
             patch.object(search, "expand_query_to_subqueries", return_value=["q"]), \
             patch.object(search, "query_collection", return_value=[candidate]), \
             patch.object(search, "lexical_search", return_value=[]), \
             patch.object(search, "rerank_candidates", side_effect=lambda q, x: x), \
             patch.object(search, "select_contexts", return_value=(selection, [candidate])), \
             patch.object(search, "generate_answer", side_effect=AssertionError("generation must not run")):
            result = search.run("q", llm=llm, client=client, model=model, quiet=True)
        self.assertIsNone(result["answer"])
        self.assertFalse(result["generation_enabled"])
        self.assertEqual(result["answer_sources"], [])
        self.assertEqual([c["chunk_id"] for c in result["selected_contexts"]], ["c1"])

    def test_generation_disabled_skips_generator_llm_client(self):
        # 关闭生成时不创建 generator 角色客户端（其余角色照常），
        # 覆盖 retriever._run 中 auto-create 分支的开关（上面的用例显式传 llm，绕过了它）。
        from pipeline.retrieval import retriever as search
        client = Mock()
        client.list_collections.return_value = ["rag_pcb"]
        client.get_collection.return_value = Mock()
        model = Mock()
        model.encode.return_value = [[1.0, 0.0]]
        candidate = {
            "chunk_id": "c1", "document": "ctx", "metadata": {"domain": "pcb"},
            "distance": 0.1, "similarity": 0.9, "retrieval_source": "vector"
        }
        selection = {
            "selected_chunk_ids": ["c1"], "sufficient": True,
            "missing_information": [], "expansion_requests": [], "reason": "test",
        }
        roles = []

        def fake_client(role=None, **kwargs):
            roles.append(role)
            return object()

        with patch.object(settings.rag.generation, "enabled", False), \
             patch.object(search, "create_llm_client", side_effect=fake_client), \
             patch.object(search, "analyze_query", return_value={"keywords": [], "domains": ["pcb"], "reason": "test"}), \
             patch.object(search, "expand_query_to_subqueries", return_value=["q"]), \
             patch.object(search, "query_collection", return_value=[candidate]), \
             patch.object(search, "lexical_search", return_value=[]), \
             patch.object(search, "rerank_candidates", side_effect=lambda q, x: x), \
             patch.object(search, "select_contexts", return_value=(selection, [candidate])), \
             patch.object(search, "generate_answer", side_effect=AssertionError("generation must not run")):
            result = search.run("q", client=client, model=model, quiet=True)
        self.assertNotIn("generator", roles)
        self.assertIn("selector", roles)
        self.assertIsNone(result["answer"])

    def test_splitter_keeps_specialized_fields(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "eval.jsonl"
            source.write_text(json.dumps(full_row(), ensure_ascii=False) + "\n", encoding="utf-8")
            manifest = split_benchmarks(source, root / "bench")
            self.assertEqual(manifest["counts"], {"router": 1, "retrieval": 1, "generation": 1, "e2e": 1})
            generation = json.loads((root / "bench" / "generation_test.jsonl").read_text("utf-8"))
            self.assertEqual(generation["reference_context_ids"], ["c1"])
            self.assertEqual(generation["question_type"], "single")

    def test_router_evaluator_writes_real_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / "router.jsonl"
            row = {"id": "q1", "user_input": "PCB?", "expected_domains": ["pcb"], "question_type": "single"}
            data.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            summary = evaluate_router(data, root / "out", llm=FakeRouterLLM())
            self.assertEqual(summary["micro"]["f1"], 1.0)
            self.assertTrue((root / "out" / "scores.csv").exists())

    def test_generation_evaluator_uses_gold_context(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / "generation.jsonl"
            row = full_row()
            keep = {k: row[k] for k in (
                "id", "user_input", "reference", "reference_context_ids", "reference_contexts",
                "reference_sources", "question_type"
            )}
            data.write_text(json.dumps(keep, ensure_ascii=False) + "\n", encoding="utf-8")
            summary = evaluate_generation(
                data, root / "out", llm=FakeAnswerLLM(), scorer=FakeGenerationScorer()
            )
            self.assertEqual(summary["metrics"]["faithfulness"]["mean"], 0.9)
            details = json.loads((root / "out" / "details.jsonl").read_text("utf-8"))
            self.assertEqual(details["gold_context_count"], 1)


if __name__ == "__main__":
    unittest.main()
