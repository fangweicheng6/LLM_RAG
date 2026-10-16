
from core.registry.component_registry import ComponentRegistry
from utils.config_loader import settings


class ComponentLoader:
    """Load v7.16 components into the unified service runtime."""

    def __init__(self, config=None):
        self.config = config or {}
        self.registry = ComponentRegistry()

    def load(self):
        from server.runtime.components import RetrieverAdapter, GeneratorAdapter
        from pipeline.routing.router import analyze_query

        self.registry.register("router", analyze_query if settings.rag.router.enabled else None)
        self.registry.register("retriever", RetrieverAdapter())
        self.registry.register("generator", GeneratorAdapter())

        try:
            from observability import trace_rag
            self.registry.register("observability", trace_rag)
        except Exception:
            self.registry.register("observability", None)

        return self.registry
