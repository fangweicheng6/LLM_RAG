# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import List

from utils.schema import CanonicalBlock
from parsers.base import BaseParser
from parsers.formula_utils import enhance_formula_text


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
                        content=enhance_formula_text(text),
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
                            content=enhance_formula_text(md),
                            section_path=" > ".join(heading_stack) or path.stem,
                        )
                    )

        # 页眉 / 页脚：常含文档编号、标题、公司名等关键标识，body 直接子元素
        # 遍历不到，单独读取，否则这些标识会从知识库中丢失。
        seen_headers = set()
        for section in doc.sections:
            header_parts = [
                section.header,
                section.footer,
                # 首页/偶页页眉是较新 python-docx 才有的属性，用 getattr 兜底，
                # 避免旧版本因缺少属性而让整页解析失败。
                getattr(section, "first_page_header", None),
                getattr(section, "even_page_header", None),
            ]
            for part in header_parts:
                if part is None:
                    continue
                try:
                    header_text = "\n".join(
                        p.text.strip()
                        for p in part.paragraphs
                        if p.text and p.text.strip()
                    )
                except Exception:
                    header_text = ""
                if header_text and header_text not in seen_headers:
                    seen_headers.add(header_text)
                    blocks.append(
                        CanonicalBlock(
                            content_type="text",
                            content=enhance_formula_text(header_text),
                            section_path=path.stem,
                        )
                    )

        # 文本框：<w:txbxContent> 嵌套在 body 的绘图/形状里，不属于直接子元素，
        # 递归查找并抽取其中的段落。
        try:
            from docx.oxml.ns import qn
            for txbx in doc.element.body.iter(qn("w:txbxContent")):
                paras = []
                for p_el in txbx.findall(qn("w:p")):
                    try:
                        t = Paragraph(p_el, doc).text.strip()
                    except Exception:
                        t = ""
                    if t:
                        paras.append(t)
                if paras:
                    blocks.append(
                        CanonicalBlock(
                            content_type="text",
                            content=enhance_formula_text("\n".join(paras)),
                            section_path=path.stem,
                        )
                    )
        except Exception:
            pass

        return blocks
