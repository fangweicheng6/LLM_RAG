# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import List

from schema import CanonicalBlock
from parsers.base import BaseParser
from parsers.common import read_text_safely


class HTMLParser(BaseParser):
    extensions = (".html", ".htm")
    document_type = "html"

    def parse(self, path: Path) -> List[CanonicalBlock]:
        try:
            from bs4 import BeautifulSoup, Tag
        except ImportError as e:
            raise RuntimeError(
                "解析 HTML 需要 beautifulsoup4。\n"
                "请执行：pip install beautifulsoup4 lxml"
            ) from e

        html = read_text_safely(path)
        soup = BeautifulSoup(html, "lxml")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        blocks: List[CanonicalBlock] = []
        heading_stack: List[str] = []

        for node in soup.find_all(
            ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "table"]
        ):
            name = node.name.lower()

            if name.startswith("h") and len(name) == 2 and name[1].isdigit():
                level = int(name[1])
                text = node.get_text(" ", strip=True)

                if not text:
                    continue

                heading_stack = heading_stack[: level - 1]
                heading_stack.append(text)
                continue

            if name == "table":
                rows = []

                for tr in node.find_all("tr"):
                    cells = [
                        c.get_text(" ", strip=True).replace("|", r"\|")
                        for c in tr.find_all(["th", "td"])
                    ]

                    if cells:
                        rows.append(cells)

                if not rows:
                    continue

                width = max(len(r) for r in rows)
                rows = [r + [""] * (width - len(r)) for r in rows]

                md_lines = [
                    "| " + " | ".join(rows[0]) + " |",
                    "| " + " | ".join(["---"] * width) + " |",
                ]

                for row in rows[1:]:
                    md_lines.append("| " + " | ".join(row) + " |")

                content = "\n".join(md_lines)
                ctype = "table"

            else:
                content = node.get_text(
                    "\n" if name == "pre" else " ",
                    strip=True,
                )
                ctype = "code" if name == "pre" else "text"

            if content:
                blocks.append(
                    CanonicalBlock(
                        content_type=ctype,
                        content=content,
                        section_path=" > ".join(heading_stack) or path.stem,
                    )
                )

        return blocks
