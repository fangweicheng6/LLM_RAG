# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I | re.S)


def _extract_first_json_value(text: str) -> str:
    text = text.strip()
    text = _FENCE_RE.sub("", text).strip()

    starts = [
        (text.find("{"), "{", "}"),
        (text.find("["), "[", "]"),
    ]
    starts = [item for item in starts if item[0] >= 0]

    if not starts:
        raise ValueError(f"LLM 未返回 JSON：{text[:300]}")

    start, opening, closing = min(starts, key=lambda x: x[0])

    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(text)):
        ch = text[idx]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                return text[start: idx + 1]

    # 返回剩余部分，交给 repair 尝试修复截断 JSON
    return text[start:]


def parse_json_response(text: str) -> Any:
    raw = _extract_first_json_value(text)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # LLM 输出可能存在：
    # 1. 缺少括号
    # 2. 截断数组
    # 3. 多余说明文字
    # 使用 json_repair 进行容错恢复。
    try:
        from json_repair import repair_json

        repaired = repair_json(raw)
        return json.loads(repaired)
    except Exception as e:
        raise ValueError(
            f"LLM 返回 JSON 无法修复：{e}\n原始内容：{raw[:500]}"
        ) from e
