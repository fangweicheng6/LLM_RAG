from server.runtime.component_loader import ComponentLoader
from core.engine.rag_engine import RAGEngine
from utils.config_loader import settings

_service=None

def create_service(config=None):
    global _service
    if _service is None:
        config = config or settings.as_dict()
        registry = ComponentLoader(config).load()
        _service = RAGEngine(registry)
        from server.runtime.preload import preload_runtime
        preload_runtime()
    return _service
