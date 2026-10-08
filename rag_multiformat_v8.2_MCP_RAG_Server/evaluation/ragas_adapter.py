"""将项目现有 Provider 和本地 embedding 接入固定版本的 Ragas。

只有新增评估入口导入本模块，旧入口无需安装 Ragas。
"""
from __future__ import annotations

import asyncio
import os
from importlib.metadata import version
from threading import Lock

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

from utils.config_loader import settings

if version("ragas") != "0.4.3":
    raise RuntimeError("本项目适配 ragas==0.4.3，请运行 python -m pip install -r requirements_eval.txt")

from langchain_core.outputs import Generation, LLMResult
from ragas.embeddings import BaseRagasEmbeddings
from ragas.llms import BaseRagasLLM
from ragas.run_config import RunConfig

from evaluation.ragas_compat import normalize_ragas_fix_output


def run_config():
    return RunConfig(timeout=settings.evaluation.runtime.timeout_sec, max_retries=settings.evaluation.runtime.max_retries,
                     max_workers=settings.evaluation.runtime.max_workers, seed=settings.evaluation.runtime.seed)


class ProjectRagasLLM(BaseRagasLLM):
    def __init__(self, client=None):
        super().__init__(run_config=run_config())
        if client is None:
            from llm.base import LLMSettings
            from llm.registry import get_provider_class
            from llm.factory import resolve_llm_model
            # 注册现有 Provider，保持 DeepSeek 等适配器行为一致。
            import llm.providers  # noqa: F401
            llm_settings = LLMSettings(
                provider=settings.evaluation.llm.provider, api_key=settings.evaluation.llm.api_key,
                base_url=settings.evaluation.llm.base_url, model=resolve_llm_model("evaluation"),
                timeout_sec=settings.evaluation.runtime.timeout_sec,
                # Ragas 负责外层重试，避免两层重试相乘。
                max_retries=0, temperature=0.0,
            )
            client = get_provider_class(llm_settings.provider)(llm_settings)
        self.client = client

    def generate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None):
        # 沿用项目 chat 接口：兼容不支持 n、response_format 的厂商。
        prompt_text = prompt.to_string()
        generations = []
        for _ in range(n):
            answer = self.client.chat(
                [{"role": "user", "content": prompt_text}],
                temperature=temperature,
                max_tokens=settings.evaluation.runtime.generation_max_tokens,
            )
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("评估模型返回空文本；已检查 content/reasoning_content/output_text")
            # Ragas 0.4.3 的 fix_output_format 输出模型是 StringIO，要求
            # {"text": "修复后的目标 JSON"}。DeepSeek/部分兼容网关会直接
            # 返回已修复的目标 JSON；只补这一层 envelope，目标 schema 仍由
            # Ragas 原生 Output Parser 继续校验，避免把真实格式错误伪装为成功。
            answer = normalize_ragas_fix_output(prompt_text, answer)
            if stop:
                for marker in stop:
                    if marker:
                        answer = answer.split(marker)[0]
            generations.append(Generation(text=answer))
        return LLMResult(generations=[generations])

    async def agenerate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None):
        return await asyncio.to_thread(self.generate_text, prompt, n, temperature, stop, callbacks)

    def is_finished(self, response):
        # 原 Provider 只返回文本；JSON 完整性由 Ragas 的 PydanticPrompt 校验。
        return bool(response.generations and all(g.text.strip() for g in response.generations[0]))


class ProjectRagasEmbeddings(BaseRagasEmbeddings):
    def __init__(self, model=None):
        super().__init__()
        self.model = model
        self.lock = Lock()
        self.set_run_config(run_config())

    def embed_documents(self, texts):
        # 不添加 retrieval query instruction：这里比较的是答案之间的语义相似度。
        with self.lock:
            if self.model is None:
                import torch
                from sentence_transformers import SentenceTransformer
                device = "cuda" if torch.cuda.is_available() else "cpu"
                # 优先离线加载本地缓存，避免访问 HuggingFace 网络超时阻塞评估流程。
                try:
                    self.model = SentenceTransformer(
                        settings.evaluation.llm.embedding_model,
                        device=device,
                        local_files_only=True,
                    )
                except Exception:
                    self.model = SentenceTransformer(
                        settings.evaluation.llm.embedding_model,
                        device=device,
                    )
            return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                                     show_progress_bar=False).tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts):
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text):
        return (await self.aembed_documents([text]))[0]


class RagasScorer:
    def __init__(self, llm=None, embeddings=None):
        # 0.4.3 的稳定兼容 API；固定版本，禁止无边界升级到 1.x。
        from ragas.metrics import LLMContextRecall, AnswerCorrectness
        llm = llm if llm is not None else ProjectRagasLLM()
        embeddings = embeddings if embeddings is not None else ProjectRagasEmbeddings()
        self.context = LLMContextRecall(llm=llm)
        self.answer = AnswerCorrectness(llm=llm, embeddings=embeddings,
                                        weights=settings.evaluation.runtime.answer_correctness_weights)
        self.context.init(run_config())
        self.answer.init(run_config())

    def score(self, row, result):
        from ragas import SingleTurnSample
        from evaluation.eval_common import checked_score
        sample = SingleTurnSample(
            user_input=row["user_input"], reference=row["reference"],
            reference_contexts=row["reference_contexts"], response=result["answer"],
            retrieved_contexts=[item["document"] for item in result["selected_contexts"]],
        )
        scores, errors = {}, {}
        for key, metric in (("context_recall", self.context), ("answer_correctness", self.answer)):
            # 空上下文不支持任何标准答案事实；空答案不可能正确。
            if (key == "context_recall" and not sample.retrieved_contexts) or (
                key == "answer_correctness" and not sample.response.strip()
            ):
                scores[key] = 0.0
                continue

            parse_attempt = 0
            while True:
                try:
                    scores[key] = checked_score(metric.single_turn_score(sample))
                    break
                except Exception as exc:
                    # TokenHub/DeepSeek 偶发返回不符合 Pydantic schema 的文本。
                    # Ragas 自身 fix_output_format 已先执行；这里只对明确的解析错误
                    # 重跑整个指标，保持原 Prompt、公式和权重不变。
                    parse_error = type(exc).__name__ in {
                        "OutputParserException", "RagasOutputParserException"
                    } or "output parser" in str(exc).lower() or "invalid json output" in str(exc).lower()
                    if parse_error and parse_attempt < settings.evaluation.runtime.metric_parse_retries:
                        parse_attempt += 1
                        continue
                    scores[key] = None
                    errors[key] = f"{type(exc).__name__}: {exc}"
                    break
        return scores, errors


class GenerationRagasScorer:
    """只评估生成器：输入使用 Gold Context，不经过 Router/Retrieval/Selector。"""
    def __init__(self, llm=None, embeddings=None):
        from ragas.metrics import AnswerCorrectness, Faithfulness, ResponseRelevancy
        llm = llm if llm is not None else ProjectRagasLLM()
        embeddings = embeddings if embeddings is not None else ProjectRagasEmbeddings()
        self.metrics = {
            "answer_correctness": AnswerCorrectness(
                llm=llm,
                embeddings=embeddings,
                weights=settings.evaluation.runtime.answer_correctness_weights,
            ),
            "faithfulness": Faithfulness(llm=llm),
            "answer_relevancy": ResponseRelevancy(llm=llm, embeddings=embeddings),
        }
        for metric in self.metrics.values():
            metric.init(run_config())

    @staticmethod
    def _is_parse_error(exc):
        return type(exc).__name__ in {
            "OutputParserException", "RagasOutputParserException"
        } or "output parser" in str(exc).lower() or "invalid json output" in str(exc).lower()

    def score(self, row, response, gold_contexts):
        from ragas import SingleTurnSample
        from evaluation.eval_common import checked_score

        sample = SingleTurnSample(
            user_input=row["user_input"],
            reference=row["reference"],
            response=response,
            retrieved_contexts=list(gold_contexts),
        )
        scores, errors = {}, {}
        for key, metric in self.metrics.items():
            if not str(response).strip():
                scores[key] = 0.0
                continue
            parse_attempt = 0
            while True:
                try:
                    scores[key] = checked_score(metric.single_turn_score(sample))
                    break
                except Exception as exc:
                    if self._is_parse_error(exc) and parse_attempt < settings.evaluation.runtime.metric_parse_retries:
                        parse_attempt += 1
                        continue
                    scores[key] = None
                    errors[key] = f"{type(exc).__name__}: {exc}"
                    break
        return scores, errors
