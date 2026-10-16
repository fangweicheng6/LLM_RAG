from server.runtime.component_loader import ComponentLoader
from core.engine.rag_engine import RAGEngine
from utils.config_loader import settings

_service=None

def create_service(config=None, preload=True):
    """创建（或复用）全局 RAG 服务。

    preload=False 时跳过模型/向量库预加载（MCP stdio 等场景需要快速完成
    初始化握手，预热改为后台线程进行）。
    """
    global _service
    if _service is None:
        config = config or settings.as_dict()
        registry = ComponentLoader(config).load()
        _service = RAGEngine(registry)
        if preload:
            from server.runtime.preload import preload_runtime
            preload_runtime()
    return _service
