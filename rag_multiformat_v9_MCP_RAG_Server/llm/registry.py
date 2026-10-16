# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, Type

from llm.base import BaseLLMClient


_PROVIDER_REGISTRY: Dict[str, Type[BaseLLMClient]] = {}
_DISCOVERED = False


def register_provider(*names: str):
    """
    Provider 注册装饰器。

    一个实现可以注册多个名字，例如：

        @register_provider("openai_compatible", "openai", "qwen")
        class OpenAICompatibleClient(...):
            ...
    """

    if not names:
        raise ValueError("register_provider 至少需要一个 provider 名称。")

    normalized = []
    for name in names:
        key = str(name).strip().lower()
        if not key:
            raise ValueError("provider 名称不能为空。")
        normalized.append(key)

    def decorator(cls: Type[BaseLLMClient]):
        if not issubclass(cls, BaseLLMClient):
            raise TypeError(
                f"Provider {cls.__name__} 必须继承 BaseLLMClient。"
            )

        for key in normalized:
            old = _PROVIDER_REGISTRY.get(key)
            if old is not None and old is not cls:
                raise ValueError(
                    f"LLM Provider 名称重复注册：{key!r}"
                )
            _PROVIDER_REGISTRY[key] = cls

        return cls

    return decorator


def discover_providers() -> None:
    """
    自动导入 llm/providers/ 下的全部 Provider 模块。

    因此新增 Provider 时只需：
        1. 在 llm/providers/ 新增一个 .py 文件；
        2. 使用 @register_provider(...) 注册；

    不再需要修改 factory.py。
    """

    global _DISCOVERED
    if _DISCOVERED:
        return

    import llm.providers as providers_package

    for module_info in pkgutil.iter_modules(
        providers_package.__path__,
        providers_package.__name__ + ".",
    ):
        module_name = module_info.name
        if module_name.endswith(".__init__"):
            continue
        importlib.import_module(module_name)

    _DISCOVERED = True


def get_provider_class(name: str) -> Type[BaseLLMClient]:
    discover_providers()

    key = str(name).strip().lower()
    if not key:
        raise ValueError("settings.model.llm.provider 不能为空。")

    provider_cls = _PROVIDER_REGISTRY.get(key)
    if provider_cls is not None:
        return provider_cls

    # 对未知名称做 OpenAI-compatible 兼容回退。
    # 这样只要目标服务提供 Chat Completions 兼容接口，
    # 用户可以使用任意 provider 名称 + 对应 BASE_URL/MODEL/KEY，
    # 无需再新增代码。
    compatible_cls = _PROVIDER_REGISTRY.get("openai_compatible")
    if compatible_cls is not None:
        return compatible_cls

    raise ValueError(
        f"没有可用的 LLM Provider：{name!r}。"
    )


def list_registered_providers() -> list[str]:
    discover_providers()
    return sorted(_PROVIDER_REGISTRY)
