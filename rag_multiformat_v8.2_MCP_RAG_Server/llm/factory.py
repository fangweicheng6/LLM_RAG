# -*- coding: utf-8 -*-
from __future__ import annotations

from utils.config_loader import settings
from llm.base import BaseLLMClient, LLMSettings
from llm.registry import get_provider_class


def resolve_llm_model(role: str | None = None) -> str:
    """按角色解析模型名。

    所有 LLM 角色的模型名统一在 ``configs/model.yaml`` 的 ``model.llm.roles`` 里配置；
    未指定角色或角色未配置时回退到 ``model.llm.model``（默认模型）。
    """
    default = settings.model.llm.model
    if not role:
        return default
    return settings.model.llm.get(f"roles.{role}", default)


def create_llm_client(role: str | None = None) -> BaseLLMClient:
    """
    根据统一配置创建 LLM Client；``role`` 只影响模型名（model.llm.roles.<role>），
    其余参数（provider / base_url / temperature / disable_thinking 等）全部共享。

    用户切换模型/厂商时，正常只需要修改四项：
        settings.model.llm.provider
        settings.model.llm.api_key
        settings.model.llm.base_url
        settings.model.llm.model  （以及可选的 settings.model.llm.roles）

    factory.py 不再包含 provider 的 if / elif 分支。
    """

    llm_settings = LLMSettings(
        provider=settings.model.llm.provider.strip().lower(),
        api_key=settings.model.llm.api_key,
        base_url=settings.model.llm.base_url,
        model=resolve_llm_model(role),
        timeout_sec=settings.model.llm.timeout_sec,
        max_retries=settings.model.llm.max_retries,
        temperature=settings.model.llm.temperature,
        disable_thinking=settings.model.llm.get("disable_thinking", True),
    )

    provider_cls = get_provider_class(
        llm_settings.provider
    )

    return provider_cls(llm_settings)
