# RAG Multiformat v7.3 Industrial Enterprise Service

## 1. 项目简介

v7.3是在v7.1多格式工业RAG基础上的企业服务化版本。

主要能力：

-   多格式知识解析（PDF/DOCX/PPT/XLSX/HTML/TXT等）
-   Router多领域路由
-   Multi-hop多跳检索
-   Reranker重排序
-   LLM Context Selector上下文筛选
-   RAGAS评估
-   Socket服务
-   HTTP API服务
-   Langfuse链路追踪接口
-   Prometheus指标监控接口
-   OpenTelemetry接口
-   知识库增量更新检测

------------------------------------------------------------------------

# 2. 环境准备

## Python环境

推荐：

Python \>= 3.10

创建虚拟环境：

``` bash
python -m venv rag
```

Windows:

``` powershell
.\rag\Scripts\activate
```

安装依赖：

``` bash
pip install -r requirements_v7.3.txt
```

如果使用完整RAG功能：

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# 3. 项目主要入口文件

## 3.1 知识库构建入口

文件：

    ingest.py

作用：

将原始文档解析、切片、embedding并建立向量索引。

执行：

``` bash
python ingest.py
```

生成：

    data_index_only/

    ├── vector_db

    ├── processed

    └── source_store

------------------------------------------------------------------------

# 4. RAG查询入口

## 方式一：直接运行

执行：

``` bash
python search.py
```

用于本地调试。

------------------------------------------------------------------------

# 5. Socket服务模式

## 5.1 启动Server

入口：

    rag_server.py

执行：

``` bash
python rag_server.py
```

默认：

    0.0.0.0:9001

启动后：

    Client
     |
    Socket(JSON)
     |
    RAG Server
     |
    RAG Pipeline

------------------------------------------------------------------------

## 5.2 Client请求

入口：

    rag_client.py

执行：

``` bash
python rag_client.py "你的问题"
```

例如：

``` bash
python rag_client.py "IGBT温升异常原因是什么"
```

------------------------------------------------------------------------

# 6. HTTP API服务

入口：

    api_server.py

启动：

``` bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

访问接口：

    http://localhost:8000/docs

Swagger页面可以直接测试。

接口：

## 查询

POST:

    /chat

请求：

``` json
{
 "question":"问题"
}
```

返回：

``` json
{
 "answer":"回答"
}
```

------------------------------------------------------------------------

# 7. 监控系统

## Prometheus

指标接口：

    /metrics

访问：

    http://localhost:8000/metrics

用于Prometheus采集。

------------------------------------------------------------------------

## Grafana

需要额外部署：

-   Prometheus Server
-   Grafana

推荐部署：

Docker Compose

------------------------------------------------------------------------

# 8. Langfuse配置

用于记录：

-   用户问题
-   Router结果
-   检索结果
-   LLM调用
-   Token消耗
-   延迟

配置环境变量：

Windows:

``` powershell
setx LANGFUSE_PUBLIC_KEY "xxx"

setx LANGFUSE_SECRET_KEY "xxx"

setx LANGFUSE_HOST "http://localhost:3000"
```

如果使用官方云：

    LANGFUSE_HOST=https://cloud.langfuse.com

------------------------------------------------------------------------

# 9. OpenTelemetry

入口：

    otel_config.py

用于统一Trace：

    Client

    ↓

    Server

    ↓

    Retriever

    ↓

    LLM

后续可连接：

-   Langfuse
-   Grafana
-   OTEL Collector

------------------------------------------------------------------------

# 10. 知识库更新

入口：

    knowledge_update.py

功能：

检测：

-   新文件
-   修改文件
-   文件hash变化

执行：

``` bash
python knowledge_update.py
```

流程：

    文件

    ↓

    Hash检测

    ↓

    变化记录

    ↓

    重新解析

    ↓

    更新索引

------------------------------------------------------------------------

# 11. 完整运行流程

推荐流程：

## 第一步：建立知识库

    python ingest.py

## 第二步：启动RAG服务

Socket:

    python rag_server.py

或者HTTP:

    uvicorn api_server:app --host 0.0.0.0 --port 8000

## 第三步：客户端访问

Socket:

    python rag_client.py "问题"

HTTP:

调用：

    POST /chat

------------------------------------------------------------------------

# 12. 工业部署建议

生产环境推荐：

    Client

    ↓

    Nginx

    ↓

    FastAPI

    ↓

    RAG Server

    ↓

    Vector Database

    ↓

    Langfuse

    ↓

    Prometheus

    ↓

    Grafana

------------------------------------------------------------------------

# 13. 版本说明

v7.3目标：

从算法型RAG升级为：

工业RAG服务平台。

已具备：

-   服务化
-   可观测性接口
-   增量知识管理
-   企业部署基础
