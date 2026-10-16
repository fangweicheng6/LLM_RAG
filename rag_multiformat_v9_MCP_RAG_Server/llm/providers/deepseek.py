# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, List

from llm.base import LLMSettings, Message, _retry_json_call
from llm.providers.openai_compatible import OpenAICompatibleClient
from llm.registry import register_provider


@register_provider("deepseek")
class DeepSeekClient(OpenAICompatibleClient):
    """
    DeepSeek Provider。

    DeepSeek 使用 OpenAI-compatible Chat Completions 形式，因此复用通用实现。
    本类只保留 DeepSeek 的 JSON 输出增强能力。

    配置仍然只有统一四项：
        settings.model.llm.provider
        settings.model.llm.api_key
        settings.model.llm.base_url
        settings.model.llm.model
    """

    def __init__(self, settings: LLMSettings):
        super().__init__(settings)

    def chat_json(
        self,
        messages: List[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        def call(msgs, *, temperature, max_tokens):
            return self._request(
                msgs,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_payload={
                    "response_format": {
                        "type": "json_object",
                    }
                },
            )

        return _retry_json_call(
            call,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
