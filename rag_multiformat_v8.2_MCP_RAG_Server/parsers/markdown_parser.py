# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from utils.schema import CanonicalBlock
from parsers.base import BaseParser
from parsers.formula_utils import enhance_formula_text
from parsers.common import read_text_safely


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^\s*(```+|~~~+)")


class MarkdownParser(BaseParser):
    extensions = (".md", ".markdown")
    document_type = "markdown"

    def parse(self, path: Path) -> List[CanonicalBlock]:
        """
        按 Markdown 章节解析。

        重要规则：
        - 标题只负责更新 section_path，不进入正文；
        - 普通文本和 fenced code 分开输出，避免代码块与后续文本粘连；
        - 同一章节可以输出多个 CanonicalBlock，后续由 clean_chunk.py 按章节合并；
        - 不在 Parser 层做 overlap。
        """
        text = read_text_safely(path)
        blocks: List[CanonicalBlock] = []

        heading_stack: List[str] = []
        text_buf: List[str] = []
        code_buf: List[str] = []

        in_fence = False
        fence_token: str | None = None

        def current_section() -> str:
            return " > ".join(heading_stack)

        def flush_text() -> None:
            nonlocal text_buf
            content = "\n".join(text_buf).strip()
            if content:
                blocks.append(
                    CanonicalBlock(
                        content_type="text",
                        content=enhance_formula_text(content),
                        section_path=current_section(),
                    )
                )
            text_buf = []

        def flush_code() -> None:
            nonlocal code_buf
            content = "\n".join(code_buf).strip()
            if content:
                blocks.append(
                    CanonicalBlock(
                        content_type="code",
                        content=enhance_formula_text(content),
                        section_path=current_section(),
                    )
                )
            code_buf = []

        for line in text.splitlines():
            fence = _FENCE_RE.match(line)

            if in_fence:
                code_buf.append(line)

                if fence:
                    token = fence.group(1)[:3]
                    if token == fence_token:
                        in_fence = False
                        fence_token = None
                        flush_code()
                continue

            # fenced code 开始前，先把普通文本落盘，避免两种内容粘在一起。
            if fence:
                flush_text()
                in_fence = True
                fence_token = fence.group(1)[:3]
                code_buf = [line]
                continue

            heading_match = _HEADING_RE.match(line)
            if heading_match:
                flush_text()

                level = len(heading_match.group(1))
                heading = re.sub(
                    r"\s+#+\s*$",
                    "",
                    heading_match.group(2),
                ).strip()

                heading_stack = heading_stack[: level - 1]
                heading_stack.append(heading)
                continue

            # 空行作为自然段边界，输出一个 text block。
            if not line.strip():
                flush_text()
                continue

            text_buf.append(line)

        # EOF：兼容未闭合的 fenced code。
        if in_fence:
            flush_code()
        else:
            flush_text()

        return blocks
