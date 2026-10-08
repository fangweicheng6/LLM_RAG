# RAG 客户端访问说明

## 1. HTTP Client

启动 HTTP 服务:

```bash
uvicorn deployment.api_server:app --host 0.0.0.0 --port 8000
```

调用:

```bash
python -m client.http_client "什么是阻抗匹配?"
```

接口:
- POST `/chat` body: `{"question":"..."}`
- GET `/health`

## 2. Socket Client

启动 TCP 服务:

```bash
python -m deployment.rag_server
```

调用:

```bash
python -m client.socket_client "什么是阻抗匹配?"
```

协议:
请求:
```json
{"cmd":"query","text":"问题"}
```

健康检查:
```json
{"cmd":"health"}
```
