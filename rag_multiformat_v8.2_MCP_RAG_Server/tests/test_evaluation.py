"""不调用外部 LLM；安装 requirements_eval.txt 后验证真实 Ragas 实现。"""
import importlib.util
import json
import math
import uuid
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from utils.config_loader import settings
from evaluation.eval_common import (checked_score, load_dataset, load_source_chunks, retrieval_metrics,
                         set_scores, summarize, validate_rows, verify_gold_in_index, write_jsonl)
from evaluation.generate_eval_dataset import balanced_chunks
from pipeline.rag_modules import choose_contexts, route_query
from evaluation.rag_valide import evaluate_rows, save_reports
from llm.providers.openai_compatible import extract_completion_text
from evaluation.ragas_compat import RAGAS_FIX_OUTPUT_MARKER, normalize_ragas_fix_output


def golden():
    return {"id": "q1", "user_input": "BUCK 的输入是多少？", "reference": "输入为 12 V。",
            "expected_domains": ["topology", "device"], "reference_context_ids": ["c1", "c2"],
            "reference_contexts": ["输入为 12 V。", "器件额定电压为 30 V。"]}


def trace():
    return {"router_domains": ["topology", "pcb"],
            "candidate_pool": [{"chunk_id": "c1", "document": "输入为 12 V。", "distance": 0.1},
                               {"chunk_id": "other", "document": "无关内容", "distance": 0.2}],
            "selected_contexts": [{"chunk_id": "c2", "document": "器件额定电压为 30 V。"}],
            "answer": "输入为 12 V。"}


class MetricTests(unittest.TestCase):
    def test_ragas_fix_output_direct_target_json_gets_stringio_envelope(self):
        direct = json.dumps({"statements": ["输入为 12 V。"]}, ensure_ascii=False)
        fixed = normalize_ragas_fix_output(RAGAS_FIX_OUTPUT_MARKER, direct)
        self.assertEqual(json.loads(fixed), {"text": direct})

    def test_ragas_fix_output_existing_stringio_is_not_double_wrapped(self):
        ready = json.dumps({"text": '{"statements":["输入为 12 V。"]}'}, ensure_ascii=False)
        self.assertEqual(normalize_ragas_fix_output(RAGAS_FIX_OUTPUT_MARKER, ready), ready)

    def test_ragas_fix_output_does_not_touch_normal_metric_response(self):
        direct = json.dumps({"statements": ["输入为 12 V。"]}, ensure_ascii=False)
        self.assertEqual(normalize_ragas_fix_output("normal statement prompt", direct), direct)

    def test_multilabel_metrics_and_stage_separation(self):
        metrics = retrieval_metrics(golden(), trace())
        self.assertEqual(metrics["router_f1"], 0.5)
        self.assertEqual(metrics["retrieval_recall"], 0.5)
        self.assertEqual(metrics["final_context_chunk_recall"], 0.5)
        self.assertEqual(set_scores(["a", "a"], ["a", "b"])["recall"], 0.5)
        self.assertEqual(set_scores([], ["a"])["f1"], 0)

    def test_disabled_modes_do_not_emit_misleading_metrics(self):
        metrics = retrieval_metrics(golden(), trace(), router_enabled=False, selector_enabled=False)
        self.assertEqual(metrics, {"retrieval_recall": 0.5})
        rows = [{**golden(), "status": "ok", "retrieval_recall": 0.5,
                 "context_recall": 0.5, "answer_correctness": 0.5}]
        summary = summarize(rows, router_enabled=False, selector_enabled=False)
        self.assertNotIn("router_f1", summary["metrics"])
        self.assertNotIn("router_f1_micro", summary)
        self.assertNotIn("final_context_chunk_recall", summary["metrics"])

    def test_recall_at_k(self):
        result = trace()
        result["candidate_pool"][1]["chunk_id"] = "c2"
        self.assertEqual(retrieval_metrics(golden(), result)["retrieval_recall"], 1)
        self.assertEqual(retrieval_metrics(golden(), result, 1)["retrieval_recall"], 0.5)

    def test_invalid_score_is_not_silently_averaged(self):
        for value in (math.nan, math.inf, -1, 2):
            with self.subTest(value=value), self.assertRaises(ValueError):
                checked_score(value)

    def test_failures_retained_and_partial_metrics_preserved(self):
        rows = [golden(), {**golden(), "id": "q2", "user_input": "第二题"}]
        scorer = Mock()
        scorer.score.return_value = ({"context_recall": 1.0, "answer_correctness": None},
                                     {"answer_correctness": "judge timeout"})
        results = evaluate_rows(rows, Mock(side_effect=[trace(), RuntimeError("API failed")]), scorer)
        self.assertEqual([r["status"] for r in results], ["metric_error", "pipeline_error"])
        summary = summarize(results)
        self.assertEqual(summary["failed"], 2)
        self.assertEqual(summary["metrics"]["context_recall"]["valid"], 1)
        self.assertIsNone(summary["metrics"]["answer_correctness"]["mean"])
        self.assertEqual(summary["router_f1_micro"], 0.5)

    def test_empty_retrieval_is_zero_not_pipeline_error(self):
        import search
        result = search._empty_result("q", {}, ["pcb"], [], "miss")
        values = retrieval_metrics(golden(), result)
        self.assertEqual(values["router_f1"], 0)
        self.assertEqual(values["retrieval_recall"], 0)

    def test_dataset_rejects_malformed_truth(self):
        for update in ({"expected_domains": []}, {"expected_domains": ["bad"]},
                       {"reference_context_ids": ["c1", "c1"]}, {"reviewed": "false"},
                       {"reference": ""}, {"reference_contexts": ["only one"]}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate_rows([{**golden(), **update}])
        with self.assertRaises(ValueError):
            validate_rows([golden(), golden()])

    def test_report_nulls_and_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            scorer = Mock()
            scorer.score.return_value = ({"context_recall": 1, "answer_correctness": 0.8}, {})
            result = evaluate_rows([golden()], lambda q: trace(), scorer)
            save_reports(directory, result, {"complete": True})
            self.assertTrue(json.loads((directory / "summary.json").read_text("utf-8"))["complete"])
            self.assertEqual(json.loads((directory / "details.jsonl").read_text("utf-8"))["id"], "q1")
            self.assertIn("BUCK", (directory / "scores.csv").read_text("utf-8-sig"))

    def test_disabled_report_omits_router_and_selector_columns(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            scorer = Mock()
            scorer.score.return_value = ({"context_recall": 1, "answer_correctness": 0.8}, {})
            result = evaluate_rows([golden()], lambda q: trace(), scorer)
            result = [{key: value for key, value in result[0].items()
                       if key not in {"router_f1", "router_counts", "final_context_chunk_recall"}}]
            save_reports(directory, result, {"complete": True,
                                              "settings": {"router_enabled": False,
                                                            "llm_chunk_selection_enabled": False}})
            summary = json.loads((directory / "summary.json").read_text("utf-8"))
            csv_header = (directory / "scores.csv").read_text("utf-8-sig").splitlines()[0]
            self.assertNotIn("router_f1", summary["metrics"])
            self.assertNotIn("router_f1_micro", summary)
            self.assertNotIn("final_context_chunk_recall", summary["metrics"])
            self.assertNotIn("router_f1", csv_header)
            self.assertNotIn("selected_chunk_recall", csv_header)
            self.assertIn("Router: OFF", (directory / "summary.txt").read_text("utf-8"))

    def test_openai_compatible_gateway_text_fallbacks(self):
        self.assertEqual(extract_completion_text({
            "choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
            "output_text": "top-level",
        }), "top-level")
        self.assertEqual(extract_completion_text({
            "choices": [{"message": {"content": "", "reasoning_content": "judge json"}}],
        }), "judge json")
        self.assertEqual(extract_completion_text({
            "choices": [{"message": {"content": [{"type": "text", "text": "part-1"},
                                                    {"type": "text", "text": "part-2"}]}}],
        }), "part-1part-2")
        with self.assertRaisesRegex(RuntimeError, "没有可用文本"):
            extract_completion_text({"choices": [{"message": {"content": None}}]})

    def test_output_cannot_pollute_index_store(self):
        with self.assertRaises(ValueError):
            write_jsonl(settings.paths.chroma_dir / "gold.jsonl", [golden()])
        with self.assertRaises(ValueError):
            write_jsonl(settings.paths.chunk_jsonl, [golden()])


class SwitchTests(unittest.TestCase):
    def test_router_disabled_never_invokes_analyzer(self):
        analyzer = Mock(side_effect=AssertionError("must not call LLM"))
        with patch.object(settings.rag.router, "enabled", False):
            result = route_query("query", None, ["topology", "device"], analyzer=analyzer)
        self.assertEqual(result["domains"], ["topology", "device"])
        analyzer.assert_not_called()

    def test_selector_disabled_no_expansion_and_distance_topk(self):
        selector = Mock(side_effect=AssertionError("must not call selector"))
        with patch.object(settings.rag.context.selector, "enabled", False), patch.object(settings.rag.retrieval, "top_k", 1):
            selection, contexts = choose_contexts(query="q", candidates=list(reversed(trace()["candidate_pool"])),
                                                  llm=None, allow_expansion=True, selector=selector)
        self.assertEqual(contexts[0]["chunk_id"], "c1")
        self.assertEqual(selection["expansion_requests"], [])
        self.assertFalse(selection["sufficient"])
        selector.assert_not_called()

    def test_all_four_combinations_use_same_search_pipeline(self):
        import search
        model = Mock()
        model.encode.return_value = [[1.0, 0.0]]
        client = Mock()
        client.list_collections.return_value = ["rag_topology", "rag_device"]
        def choose(**kwargs):
            return {"sufficient": True, "expansion_requests": []}, kwargs["candidates"][:1]
        for router_on in (False, True):
            for selector_on in (False, True):
                with self.subTest(router=router_on, selector=selector_on), \
                     patch.object(settings.rag.router, "enabled", router_on), patch.object(settings.rag.context.selector, "enabled", selector_on), \
                     patch.object(search, "analyze_query", return_value={"domains": ["topology"], "keywords": []}) as router, \
                     patch.object(search, "query_collection", return_value=trace()["candidate_pool"]) as query, \
                     patch.object(search, "select_contexts", side_effect=choose) as selector, \
                     patch.object(search, "expand_context_requests") as expansion, \
                     patch.object(search, "generate_answer", return_value=("answer", [])):
                    result = search.run("问题", client=client, model=model, llm=object(), quiet=True)
                    self.assertEqual(router.call_count, int(router_on))
                    self.assertEqual(selector.call_count, int(selector_on))
                    self.assertEqual(query.call_count, 1 if router_on else 2)
                    self.assertEqual(result["router_enabled"], router_on)
                    self.assertTrue(result["selected_contexts"])
                    expansion.assert_not_called()

    def test_unavailable_route_preserves_prediction_for_f1(self):
        import search
        client = Mock()
        client.list_collections.return_value = ["rag_topology"]
        with patch.object(settings.rag.router, "enabled", True), \
             patch.object(search, "analyze_query", return_value={"domains": ["device"], "keywords": []}):
            response = search.run("q", client=client, llm=object(), allow_empty=True, quiet=True)
        self.assertEqual(response["router_domains"], ["device"])
        self.assertEqual(response["candidate_pool"], [])


class SourceDatasetTests(unittest.TestCase):
    def test_source_dataset_and_stale_index_detection(self):
        import clean_chunk
        import source_store
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            kb = root / "kb"
            (kb / "topology").mkdir(parents=True)
            path = kb / "topology" / "buck.txt"
            path.write_text("BUCK 的输入为 12 V，输出为 5 V，开关频率为 100 kHz。", encoding="utf-8")
            with patch.object(settings.paths, "source_store_dir", root / "sources"):
                chunks = clean_chunk.build_chunks_for_file(path, kb, 2000)
                manifest = root / "index.jsonl"
                write_jsonl(manifest, [c.to_index_dict() for c in chunks])
                items = load_source_chunks(manifest)
                self.assertEqual(items[0]["document"], chunks[0].content)
                row = {**golden(), "reference_context_ids": [chunks[0].chunk_id],
                       "reference_contexts": [chunks[0].content]}
                client = Mock()
                client.list_collections.return_value = ["rag_topology"]
                collection = client.get_collection.return_value
                collection.metadata = {"index_schema_version": settings.paths.index_schema_version}
                collection.get.return_value = {"ids": [chunks[0].chunk_id],
                                               "metadatas": [chunks[0].chroma_metadata()]}
                self.assertEqual(verify_gold_in_index([row], client), 1)
                with self.assertRaisesRegex(ValueError, "正文"):
                    verify_gold_in_index([{**row, "reference_contexts": ["tampered"]}], client)
                with self.assertRaisesRegex(ValueError, "不在当前向量库"):
                    collection.get.return_value = {"ids": [], "metadatas": []}
                    verify_gold_in_index([row], client)

    def test_balanced_sampling_is_reproducible(self):
        items = [{"chunk_id": f"c{i}", "document": "可供生成的数据内容" * 5,
                  "metadata": {"domain": "device" if i < 3 else "topology", "document_id": str(i // 2)}}
                 for i in range(8)]
        a, b = balanced_chunks(items, 4, 42), balanced_chunks(list(reversed(items)), 4, 42)
        self.assertEqual(a, b)
        self.assertEqual({item["metadata"]["domain"] for item in a[:2]}, {"device", "topology"})


class ScriptedClient:
    """仅测试用模型响应；实际运行 Ragas 的 schema、执行器和指标公式。"""
    def chat(self, messages, **kwargs):
        prompt = messages[0]["content"]
        raw = prompt.rsplit("\ninput:", 1)[1].rsplit("Output:", 1)[0].strip()
        data = json.loads(raw)
        if "source_chunk" in data:
            return json.dumps({"answerable": True, "question": "BUCK 的输入电压是多少？",
                               "reference": "输入为 12 V。", "evidence_quote": "输入为 12 V"}, ensure_ascii=False)
        if "context" in data:
            return json.dumps({"classifications": [{"statement": data["answer"], "reason": "source supports",
                                                    "attributed": 1 if "12 V" in data["context"] else 0}]})
        if isinstance(data.get("answer"), str):
            return json.dumps({"statements": [data["answer"]]})
        if "ground_truth" in data:
            equal = data["answer"] == data["ground_truth"]
            return json.dumps({"TP": [{"statement": "match", "reason": "same"}] if equal else [],
                               "FP": [] if equal else [{"statement": "wrong", "reason": "different"}],
                               "FN": [] if equal else [{"statement": "missing", "reason": "different"}]})
        raise AssertionError("unexpected prompt")


@unittest.skipUnless(importlib.util.find_spec("ragas"), "需要 requirements_eval.txt")
class RagasIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("chromadb"), "需要 chromadb")
    def test_both_cli_entries_real_chroma_ragas_and_all_switches(self):
        import chromadb
        import numpy as np
        import clean_chunk
        import source_store
        import search
        import generate_eval_dataset
        import rag_valide
        import ragas_adapter
        from utils.index_audit import assert_index_collection
        prefix = "eval_" + uuid.uuid4().hex + "_"
        client = chromadb.EphemeralClient()
        collection = client.create_collection(prefix + "topology", embedding_function=None,
                                              metadata={"index_schema_version": settings.paths.index_schema_version,
                                                        "hnsw:space": "cosine"})
        self.addCleanup(client.delete_collection, prefix + "topology")
        class Vectors:
            def encode(self, texts, **kwargs):
                return np.array([[1., 0.] for _ in texts])
        class RagClient:
            def chat(self, messages, **kwargs):
                return "输入为 12 V。"
            def chat_json(self, messages, **kwargs):
                prompt = messages[-1]["content"]
                if "selected_chunk_ids" in prompt:
                    return {"selected_chunk_ids": [chunk_id], "sufficient": True}
                if '"keywords"' in prompt:
                    return {"keywords": ["BUCK"]}
                return {"domains": ["topology"], "reason": "fixture"}
        adapter_llm = ragas_adapter.ProjectRagasLLM(ScriptedClient())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            kb = root / "kb"
            (kb / "topology").mkdir(parents=True)
            source = kb / "topology" / "buck.txt"
            source.write_text("BUCK 输入为 12 V，输出为 5 V，开关频率为 100 kHz。", encoding="utf-8")
            dataset, manifest = root / "dataset.jsonl", root / "index.jsonl"
            with patch.object(settings.paths, "source_store_dir", root / "sources"), \
                 patch.object(settings.paths, "collection_prefix", prefix), \
                 patch.object(settings.paths, "collection_prefix", prefix), \
                 patch.object(ragas_adapter, "ProjectRagasLLM", return_value=adapter_llm), \
                 patch.object(search, "get_client", return_value=client), \
                 patch.object(search, "load_embedding_model", return_value=Vectors()), \
                 patch("llm.create_llm_client", return_value=RagClient()):
                chunks = clean_chunk.build_chunks_for_file(source, kb, 2000)
                chunk_id = chunks[0].chunk_id
                write_jsonl(manifest, [chunk.to_index_dict() for chunk in chunks])
                collection.upsert(ids=[chunk_id], embeddings=[[1., 0.]],
                                  metadatas=[chunks[0].chroma_metadata()])
                self.assertEqual(generate_eval_dataset.main([
                    "--size", "1", "--index", str(manifest), "--output", str(dataset)]), 0)
                self.assertEqual(rag_valide.main(["--dataset", str(dataset), "--check-only"]), 0)
                for router_on in (True, False):
                    for selector_on in (True, False):
                        report = root / f"report_{router_on}_{selector_on}"
                        with patch.object(settings.rag.router, "enabled", router_on), \
                             patch.object(settings.rag.context.selector, "enabled", selector_on):
                            self.assertEqual(rag_valide.main([
                                "--dataset", str(dataset), "--output-dir", str(report)]), 0)
                        summary = json.loads((report / "summary.json").read_text("utf-8"))
                        self.assertTrue(summary["complete"])
                        self.assertEqual(summary["failed"], 0)
                        if router_on:
                            self.assertEqual(summary["router_f1_micro"], 1)
                        else:
                            self.assertNotIn("router_f1_micro", summary)
                        if selector_on:
                            self.assertEqual(summary["metrics"]["final_context_chunk_recall"]["mean"], 1)
                        else:
                            self.assertNotIn("final_context_chunk_recall", summary["metrics"])
                        for name in ("retrieval_recall", "context_recall", "answer_correctness"):
                            self.assertEqual(summary["metrics"][name]["mean"], 1)
                # 生成、评估结束后，数据库仍无正文。
                self.assertEqual(assert_index_collection(collection, scan=True), 1)

    def test_real_ragas_testset_generator_retains_exact_chunk_ids(self):
        from evaluation.generate_eval_dataset import generate
        from ragas_adapter import ProjectRagasLLM, ProjectRagasEmbeddings
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "golden.jsonl"
            index = Path(temp) / "index.jsonl"
            index.write_text("fixture", encoding="utf-8")
            items = [{"chunk_id": "real-chunk-7", "document": "BUCK 输入为 12 V，输出为 5 V。",
                      "metadata": {"domain": "topology", "source_ref": "a" * 64}}]
            with patch.object(settings.evaluation.runtime, "max_retries", 0):
                rows = generate(items, size=1, output=output, seed=42, batch_size=1, index_path=index,
                                llm=ProjectRagasLLM(ScriptedClient()), embeddings=ProjectRagasEmbeddings())
            self.assertEqual(load_dataset(output), rows)
            self.assertEqual(rows[0]["reference_context_ids"], ["real-chunk-7"])
            self.assertEqual(rows[0]["expected_domains"], ["topology"])
            self.assertFalse(rows[0]["reviewed"])
            self.assertFalse(output.with_suffix(".partial.jsonl").exists())

    def test_real_ragas_metrics_with_deterministic_provider(self):
        import numpy as np
        from ragas_adapter import ProjectRagasLLM, ProjectRagasEmbeddings, RagasScorer
        class Vectors:
            def encode(self, texts, **kwargs):
                return np.array([[1., 0.] if "12 V" in t else [0., 1.] for t in texts])
        scorer = RagasScorer(ProjectRagasLLM(ScriptedClient()), ProjectRagasEmbeddings(Vectors()))
        response = trace()
        response["selected_contexts"] = [{"document": "输入为 12 V。"}]
        values, errors = scorer.score(golden(), response)
        self.assertEqual(errors, {})
        self.assertEqual(values, {"context_recall": 1, "answer_correctness": 1})
        response["answer"] = "输入为 48 V。"
        response["selected_contexts"] = [{"document": "输入为 48 V。"}]
        values, errors = scorer.score(golden(), response)
        self.assertEqual(errors, {})
        self.assertEqual(values, {"context_recall": 0, "answer_correctness": 0})

    def test_empty_context_does_not_call_context_judge(self):
        from ragas_adapter import ProjectRagasLLM, ProjectRagasEmbeddings, RagasScorer
        scorer = RagasScorer(ProjectRagasLLM(ScriptedClient()), ProjectRagasEmbeddings())
        response = {"answer": "", "selected_contexts": []}
        self.assertEqual(scorer.score(golden(), response), ({"context_recall": 0, "answer_correctness": 0}, {}))


if __name__ == "__main__":
    unittest.main()
