# v7.13 四类 Benchmark 使用说明

## 1. 本次修复的核心问题

v7.12 虽然生成了 `router_test.jsonl`、`retrieval_test.jsonl`、`generation_test.jsonl` 和 `e2e_test.jsonl`，但只有 E2E 实际调用了评估器；Router / Retrieval / Generation 只是打印“可接入评估器”。另外 E2E 入口没有显式传入 `e2e_test.jsonl`，实际仍评估完整 `eval_dataset.jsonl`。

当前版本把四套 benchmark 接通，并将运行配置统一收敛到 `configs/*.yaml`；Router 与 Context Selector 的开关不再由 Python 常量覆盖。

## 2. 一次运行全部

```powershell
python run_benchmark_evaluation.py --type all
```

程序会先刷新：

```text
data_index_only/dataset/benchmarks/
├── benchmark_manifest.json
├── router_test.jsonl
├── retrieval_test.jsonl
├── generation_test.jsonl
└── e2e_test.jsonl
```

再分别运行四套评估。

## 3. 四类指标

### Router

只测试 Domain Router，不跑向量检索和答案生成：

- Precision (micro)
- Recall (micro)
- F1 (micro)
- Macro Precision / Recall / F1
- Exact Match Accuracy

### Retrieval

完整复用实际检索链：Router -> Multi-hop -> Embedding -> Chroma -> Dedup -> Reranker，并在此停止，不调用 LLM Chunk Selector 和 Answer Generator。

默认输出：

- Candidate Pool Recall
- MRR
- Recall@1 / @3 / @5 / @10
- Precision@1 / @3 / @5 / @10
- Hit Rate@1 / @3 / @5 / @10

可自定义：

```powershell
python run_benchmark_evaluation.py --type retrieval --retrieval-ks 1,3,5,10
```

注意：如果 `RERANKER_ENABLED=True` 且 `RERANK_TOP_K=6`，那么 `@10` 实际最多只能看到 reranker 输出的 6 个候选；该限制会写入 Retrieval summary 的 settings。

### Generation

直接把 Gold Context 喂给 Answer Generator，完全绕过 Router / Retrieval / Selector。因此它回答的是：**在上下文本身正确的情况下，生成模型和 Prompt 能不能答好。**

指标：

- Answer Correctness
- Faithfulness
- Answer Relevancy

这三项使用固定 `ragas==0.4.3`。

### E2E

使用 `e2e_test.jsonl` 跑完整 RAG：

- Router F1 (micro)
- Retrieval Recall
- Context Recall
- Final Context Chunk Recall（Selector 开启时）
- Answer Correctness

默认 E2E 只纳入 `knowledge_qa / reasoning / comparison` 三类 intent；当前 100 条数据会得到 85 条 E2E 样本。该范围可由环境变量 `RAG_BENCHMARK_E2E_INTENTS` 调整。

## 4. 输出目录

```text
data_index_only/dataset/benchmark_results/run_YYYYMMDD_HHMMSS_xxxxxx/
├── router/
│   ├── summary.json
│   ├── summary.txt
│   ├── details.jsonl
│   └── scores.csv
├── retrieval/
│   ├── summary.json
│   ├── summary.txt
│   ├── details.jsonl
│   └── scores.csv
├── generation/
│   ├── summary.json
│   ├── summary.txt
│   ├── details.jsonl
│   └── scores.csv
├── e2e/
│   ├── summary.json
│   ├── summary.txt
│   ├── details.jsonl
│   └── scores.csv
├── benchmark_summary.json
└── benchmark_summary.txt
```

## 5. 建议先做 5 条联调

```powershell
python -c "from utils.config_loader import settings; print('Router =', settings.rag.router.enabled); print('Selector =', settings.rag.context.selector.enabled)"
```

应看到：

```text
Router = True
Selector = True
```

然后：

```powershell
python run_benchmark_evaluation.py --type all --limit 5
```

确认四类评估均能调用模型后，再运行完整 100 / 85 条。

## 6. 评估依赖

若 Ragas 尚未安装：

```powershell
pip install -r requirements_eval.txt
```

项目内已经固定携带：

```text
third_party/ragas-0.4.3-py3-none-any.whl
```

因此不会因为 Ragas 大版本升级导致指标接口漂移。
