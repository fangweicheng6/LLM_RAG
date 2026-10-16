# -*- coding: utf-8 -*-
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List


Message = Dict[str, str]


@dataclass(frozen=True)
class LLMSettings:
    """
    所有 Provider 共用的统一配置。

    用户真正需要切换的只有：
    - provider
    - api_key
    - base_url
    - model

    timeout / retries / temperature 属于框架运行参数，通常无需修改。
    """

    provider: str
    api_key: str
    base_url: str
    model: str
    timeout_sec: int = 90
    max_retries: int = 2
    temperature: float = 0.0
    disable_thinking: bool = True


def _retry_json_call(
    call,
    messages: List[Message],
    *,
    temperature: float | None,
    max_tokens: int | None,
    retries: int = 1,
) -> Any:
    """调用 LLM 并解析 JSON；解析失败时追加“只输出合法 JSON”重试一次。

    部分模型即使被要求输出 JSON，仍偶发返回解释文字 / 截断 JSON / 代码块。
    单次失败不应直接让 Router/Selector 降级，这里做一次轻量重试。
    """
    from llm.json_utils import parse_json_response

    last_error: Exception | None = None
    attempt_messages = list(messages)

    for attempt in range(retries + 1):
        text = call(
            attempt_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            return parse_json_response(text)
        except ValueError as exc:
            last_error = exc
            if attempt >= retries:
                break
            attempt_messages = list(messages) + [
                {
                    "role": "user",
                    "content": (
                        "你上一条输出不是合法 JSON。请只输出一个合法 JSON"
                        "（对象或数组），不要 Markdown 代码块、不要任何解释文字。"
                    ),
                }
            ]

    raise last_error


class BaseLLMClient(ABC):
    """LLM 统一调用接口。Router / Selector / Answer 不感知具体厂商。"""

    def __init__(self, settings: LLMSettings):
        self.settings = settings

    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        raise NotImplementedError

    def chat_json(
        self,
        messages: List[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        return _retry_json_call(
            self.chat,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
