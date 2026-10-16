# -*- coding: utf-8 -*-
"""
Final Answer Guard

对 Answer LLM 的原始输出做轻量、确定性的质量保护：
1. 删除模型自行生成的 Sources / 参考来源尾部，来源由程序统一打印；
2. 截断明显的尾部循环重复；
3. 不改写公式、正文事实或引用编号。

该模块不是 Reranker，也不会重新调用 LLM。
"""
from __future__ import annotations

import re
from typing import List


_SOURCE_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:sources?|references?|参考来源|参考文献|参考资料|来源)\s*[:：]?\s*$",
    re.IGNORECASE,
)


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _strip_model_source_appendix(text: str) -> str:
    """
    删除模型自行追加的 Sources / 参考来源部分。

    程序会根据 selected_contexts 统一打印可信来源，避免模型自己生成一份
    Sources 后，print_final_answer() 又再打印一份，造成重复。
    """
    lines = text.splitlines()
    in_fence = False

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue

        if not in_fence and _SOURCE_HEADING_RE.match(stripped):
            # 只有在已经生成了一定正文后才把它视为来源附录，避免误伤正文标题。
            prefix = "\n".join(lines[:idx]).strip()
            if len(prefix) >= 60:
                return prefix

    return text


def _meaningful_lines(text: str) -> List[tuple[int, str]]:
    """返回 (原始行号, 规范化行)；过滤空行、纯分隔线等。"""
    result: List[tuple[int, str]] = []
    in_fence = False

    for idx, raw in enumerate(text.splitlines()):
        stripped = raw.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            # 代码块内部可能天然重复，不用于循环检测。
            continue
        if not stripped:
            continue
        if re.fullmatch(r"[-_=*]{3,}", stripped):
            continue

        normalized = re.sub(r"\s+", " ", stripped)
        result.append((idx, normalized))

    return result


def _truncate_pathological_tail_loop(text: str) -> str:
    """
    检测明显的尾部循环生成。

    使用连续 3 个“有意义文本行”的精确 n-gram。
    只有第二次出现位于回答后 55% 区域时才截断，避免把正常的前文/小结
    重述误判为循环。
    """
    raw_lines = text.splitlines()
    meaningful = _meaningful_lines(text)

    if len(meaningful) < 10:
        return text

    seen: dict[tuple[str, str, str], int] = {}
    total_raw = max(len(raw_lines), 1)

    for pos in range(len(meaningful) - 2):
        raw_idx = meaningful[pos][0]
        gram = (
            meaningful[pos][1],
            meaningful[pos + 1][1],
            meaningful[pos + 2][1],
        )

        # 太短的三行组合信息量不足，不作为循环证据。
        if sum(len(x) for x in gram) < 80:
            continue

        first_raw_idx = seen.get(gram)
        if first_raw_idx is None:
            seen[gram] = raw_idx
            continue

        # 第二次重复必须已经进入答案后半段，并且与第一次相隔足够远。
        if raw_idx / total_raw >= 0.55 and raw_idx - first_raw_idx >= 5:
            truncated = "\n".join(raw_lines[:raw_idx]).rstrip()
            if len(truncated) >= 120:
                return truncated

    return text


def sanitize_answer(text: str) -> str:
    """
    对最终答案做非生成式后处理。

    不做语义改写，只删除模型自己的来源附录与明显循环尾巴。
    """
    cleaned = _normalize_newlines(text)
    cleaned = _strip_model_source_appendix(cleaned)
    cleaned = _truncate_pathological_tail_loop(cleaned)
    return cleaned.strip()
