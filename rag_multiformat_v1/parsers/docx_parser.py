# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import List

from schema import CanonicalBlock
from parsers.base import BaseParser


def _table_to_markdown(table) -> str:
    rows = []

    for row in table.rows:
        rows.append(
            [
                cell.text.strip().replace("\n", " ").replace("|", r"\|")
                for cell in row.cells
            ]
        )

    if not rows:
        return ""

    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]

    lines = [
        "| " + " | ".join(rows[0]) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]

    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


class DOCXParser(BaseParser):
    extensions = (".docx",)
    document_type = "word"

    def parse(self, path: Path) -> List[CanonicalBlock]:
        try:
            from docx import Document
            from docx.table import Table
            from docx.text.paragraph import Paragraph
            from docx.oxml.text.paragraph import CT_P
            from docx.oxml.table import CT_Tbl
        except ImportError as e:
            raise RuntimeError(
                "解析 DOCX 需要 python-docx。\n"
                "请执行：pip install python-docx"
            ) from e

        doc = Document(path)
        blocks: List[CanonicalBlock] = []

        heading_stack: List[str] = []

        for child in doc.element.body.iterchildren():
            if isinstance(child, CT_P):
                item = Paragraph(child, doc)
                text = item.text.strip()

                if not text:
                    continue

                style_name = (item.style.name or "").lower()

                if style_name.startswith("heading"):
                    try:
                        level = int(style_name.replace("heading", "").strip())
                    except ValueError:
                        level = 1

                    heading_stack = heading_stack[: level - 1]
                    heading_stack.append(text)
                    continue

                blocks.append(
                    CanonicalBlock(
                        content_type="text",
                        content=text,
                        section_path=" > ".join(heading_stack) or path.stem,
                    )
                )

            elif isinstance(child, CT_Tbl):
                table = Table(child, doc)
                md = _table_to_markdown(table)

                if md:
                    blocks.append(
                        CanonicalBlock(
                            content_type="table",
                            content=md,
                            section_path=" > ".join(heading_stack) or path.stem,
                        )
                    )

        return blocks
