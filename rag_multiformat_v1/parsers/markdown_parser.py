# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from schema import CanonicalBlock
from parsers.base import BaseParser
from parsers.common import read_text_safely


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^\s*(```+|~~~+)")


class MarkdownParser(BaseParser):
    extensions = (".md", ".markdown")
    document_type = "markdown"

    def parse(self, path: Path) -> List[CanonicalBlock]:
        text = read_text_safely(path)
        blocks: List[CanonicalBlock] = []

        heading_stack: List[str] = []
        buf: List[str] = []

        in_fence = False
        fence_token = None

        def current_section() -> str:
            return " > ".join(heading_stack)

        def flush():
            nonlocal buf
            content = "\n".join(buf).strip()
            if content:
                ctype = "code" if content.lstrip().startswith(("```", "~~~")) else "text"
                blocks.append(
                    CanonicalBlock(
                        content_type=ctype,
                        content=content,
                        section_path=current_section(),
                    )
                )
            buf = []

        for line in text.splitlines():
            fence = _FENCE_RE.match(line)

            if fence:
                token = fence.group(1)[:3]

                if not in_fence:
                    in_fence = True
                    fence_token = token
                elif token == fence_token:
                    in_fence = False
                    fence_token = None

                buf.append(line)
                continue

            if not in_fence:
                m = _HEADING_RE.match(line)

                if m:
                    flush()

                    level = len(m.group(1))
                    heading = re.sub(r"\s+#+\s*$", "", m.group(2)).strip()

                    heading_stack = heading_stack[: level - 1]
                    heading_stack.append(heading)
                    continue

                if not line.strip():
                    flush()
                    continue

            buf.append(line)

        flush()
        return blocks
