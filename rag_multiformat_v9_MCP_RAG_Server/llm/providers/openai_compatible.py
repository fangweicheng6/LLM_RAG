# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from typing import Any, List

import requests

from llm.base import BaseLLMClient, LLMSettings, Message
from llm.registry import register_provider


# 只有明确支持 `thinking` 字段的端点才注入它来关闭思考；其余 OpenAI 兼容厂商
# （qwen / glm / kimi / siliconflow 等）不认识该字段，盲目注入可能被拒或静默忽略。
_THINKING_FIELD_PROVIDERS = frozenset({"deepseek", "openai_compatible", "openai"})


def _text_from_content(value: Any) -> str:
    """Normalize OpenAI-compatible string/part-array content.

    Some gateways return empty ``message.content`` while putting the actual
    JSON/text in ``reasoning_content`` or ``output_text``.
    """
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                for key in ("text", "content", "value"):
                    text = _text_from_content(item.get(key))
                    if text:
                        parts.append(text)
                        break
        return "".join(parts).strip()
    if isinstance(value, dict):
        for key in ("text", "content", "value"):
            text = _text_from_content(value.get(key))
            if text:
                return text
    return ""


def extract_completion_text(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise RuntimeError("LLM 响应缺少 choices[0]")
    choice = choices[0]
    message = choice.get("message") or {}
    if not isinstance(message, dict):
        raise RuntimeError("LLM 响应 choices[0].message 不是对象")
    # 真正“答案”只认 content / output_text。reasoning_content 是推理模型的
    # 思考过程（thinking），绝不能当答案返回，否则：
    #   - JSON 子任务（Router/Selector/multi_hop）会拿到思考文本而解析失败；
    #   - 最终回答会泄漏整段思考过程。
    for key in ("content", "output_text"):
        text = _text_from_content(message.get(key))
        if text:
            return text
    # 少数网关把答案放在顶层 choice/data，而不是 message 内。
    for key in ("output_text", "content"):
        text = _text_from_content(choice.get(key) or data.get(key))
        if text:
            return text

    finish_reason = choice.get("finish_reason") or "unknown"
    reasoning = _text_from_content(message.get("reasoning_content"))
    if reasoning:
        raise RuntimeError(
            "模型只返回了思考过程(reasoning_content)，未生成最终答案(content 为空)。"
            "通常是 max_tokens 不足（思考占用了输出预算）或未关闭思考模式。"
            f"finish_reason={finish_reason}"
        )
    raise RuntimeError(
        "LLM 响应没有可用文本（content/output_text 均为空），"
        f"finish_reason={finish_reason}"
    )


@register_provider(
    "openai_compatible",
    "openai",
    "qwen",
    "dashscope",
    "aliyun",
    "moonshot",
    "kimi",
    "zhipu",
    "glm",
    "siliconflow",
    "openrouter",
    "doubao",
    "volcengine",
    "baichuan",
    "minimax",
)
class OpenAICompatibleClient(BaseLLMClient):
    """
    通用 OpenAI-compatible Chat Completions Provider。

    只要厂商提供兼容接口：
        POST {settings.model.llm.base_url}/chat/completions

    就只需要修改：
        settings.model.llm.provider
        settings.model.llm.api_key
        settings.model.llm.base_url
        settings.model.llm.model

    无需修改 Router / Search / Context Selector。
    """

    def __init__(self, settings: LLMSettings):
        super().__init__(settings)

        if not settings.base_url.strip():
            raise ValueError("settings.model.llm.base_url 不能为空。")

        if not settings.model.strip():
            raise ValueError("settings.model.llm.model 不能为空。")

        self.api_key = settings.api_key.strip()
        self.base_url = settings.base_url.rstrip("/")
        self.model = settings.model.strip()

    def _request(
        self,
        messages: List[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_payload: dict | None = None,
    ) -> str:
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": (
                self.settings.temperature
                if temperature is None
                else temperature
            ),
            "stream": False,
        }

        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)

        if extra_payload:
            payload.update(extra_payload)

        # 推理模型（如 deepseek-v4-flash）默认会在 content 之外再产出
        # reasoning_content（思考过程）。思考会占用 max_tokens 预算，导致
        # JSON 子任务拿不到 JSON、最终回答泄漏思考过程。这里统一关闭。
        if self.settings.disable_thinking and self.settings.provider in _THINKING_FIELD_PROVIDERS:
            payload["thinking"] = {"type": "disabled"}

        last_error: Exception | None = None

        for attempt in range(self.settings.max_retries + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.settings.timeout_sec,
                )

                if (
                    400 <= response.status_code < 500
                    and response.status_code != 429
                ):
                    raise RuntimeError(
                        "LLM API 请求失败："
                        f"HTTP {response.status_code} - "
                        f"{response.text[:800]}"
                    )

                response.raise_for_status()
                data = response.json()

                return extract_completion_text(data)

            except Exception as exc:
                last_error = exc

                if (
                    isinstance(exc, RuntimeError)
                    and "HTTP 4" in str(exc)
                    and "HTTP 429" not in str(exc)
                ):
                    break

                if attempt >= self.settings.max_retries:
                    break

                time.sleep(1.5 * (attempt + 1))

        raise RuntimeError(
            "LLM 调用失败："
            f"provider={self.settings.provider}, "
            f"model={self.model}, "
            f"base_url={self.base_url}: {last_error}"
        ) from last_error

    def chat(
        self,
        messages: List[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return self._request(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
