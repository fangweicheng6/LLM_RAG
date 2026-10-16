# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import statistics
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple

from utils.schema import CanonicalBlock
from parsers.base import BaseParser
from parsers.formula_utils import enhance_formula_text
from utils.config_loader import settings
from parsers.ocr_utils import ocr_text


def _bbox_overlap_ratio(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)

    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0

    inter = (ix1 - ix0) * (iy1 - iy0)
    area = max((ax1 - ax0) * (ay1 - ay0), 1e-6)

    return inter / area


def _table_to_markdown(rows: List[List[Any]]) -> str:
    if not rows:
        return ""

    width = max(len(r) for r in rows)
    normalized = []

    for row in rows:
        vals = []
        for value in row:
            text = "" if value is None else str(value)
            text = text.replace("\n", " ").replace("|", r"\|").strip()
            vals.append(text)

        vals += [""] * (width - len(vals))
        normalized.append(vals)

    if not normalized:
        return ""

    header = normalized[0]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]

    for row in normalized[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _is_bold(span: Dict[str, Any]) -> bool:
    font = str(span.get("font", "")).lower()
    flags = int(span.get("flags", 0))

    return (
        "bold" in font
        or "black" in font
        or bool(flags & 16)
    )


class PDFParser(BaseParser):
    """
    增强版 PDF Parser：

    1. PyMuPDF 读取文本布局，而不是整页直接 get_text("text")
    2. 根据字体大小 / 粗体启发式识别标题
    3. 保留 page 与 section_path
    4. 尝试 page.find_tables() 抽取表格
    5. 跳过与已识别表格高度重叠的正文，减少重复
    6. 扫描页 / 文本层过少时自动 OCR fallback

    仍然保持输出 CanonicalBlock，因此后面的 Chunk / Embedding / Search
    完全不需要知道 PDF 的解析细节。
    """

    extensions = (".pdf",)
    document_type = "pdf"

    def _scan_font_statistics(self, doc) -> Tuple[float, List[float]]:
        span_sizes: List[float] = []

        for page in doc:
            try:
                data = page.get_text("dict", sort=True)
            except TypeError:
                data = page.get_text("dict")

            for block in data.get("blocks", []):
                if block.get("type") != 0:
                    continue

                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = str(span.get("text", "")).strip()

                        if len(text) < 2:
                            continue

                        try:
                            size = float(span.get("size", 0.0))
                        except Exception:
                            continue

                        if size > 0:
                            span_sizes.append(size)

        if not span_sizes:
            return 10.0, []

        body_size = statistics.median(span_sizes)

        heading_sizes = sorted(
            {
                round(size, 1)
                for size in span_sizes
                if size >= body_size * 1.18
            },
            reverse=True,
        )

        return body_size, heading_sizes[:6]

    def _heading_level(
        self,
        line_text: str,
        max_size: float,
        is_bold: bool,
        body_size: float,
        heading_sizes: List[float],
    ) -> int | None:
        text = line_text.strip()

        if not text:
            return None

        if len(text) > 120:
            return None

        # 过长、明显像普通句子的行尽量不当标题
        if len(text) > 55 and text.endswith(("。", "；", ";", ".", "，", ",")):
            return None

        strong_size = max_size >= body_size * 1.18
        bold_heading = is_bold and max_size >= body_size * 1.08

        if not (strong_size or bold_heading):
            return None

        if not heading_sizes:
            return 1

        nearest = min(
            range(len(heading_sizes)),
            key=lambda i: abs(heading_sizes[i] - max_size),
        )

        return min(nearest + 1, 6)

    def _extract_tables(self, page) -> List[Tuple[Tuple[float, float, float, float], str]]:
        if not settings.ingestion.pdf.extract_tables:
            return []

        try:
            finder = page.find_tables()
        except Exception:
            return []

        result = []

        for table in getattr(finder, "tables", []) or []:
            try:
                rows = table.extract()
                md = _table_to_markdown(rows)

                if not md:
                    continue

                bbox = tuple(float(x) for x in table.bbox)
                result.append((bbox, md))

            except Exception:
                continue

        return result

    def _ocr_page(self, page, page_number: int, section_path: str) -> List[CanonicalBlock]:
        if not settings.ingestion.ocr.enabled:
            return []

        try:
            import fitz
        except ImportError:
            return []

        matrix = fitz.Matrix(settings.ingestion.pdf.ocr_zoom, settings.ingestion.pdf.ocr_zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            pix.save(str(tmp_path))
            text = ocr_text(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        if not text.strip():
            return []

        return [
            CanonicalBlock(
                content_type="ocr",
                content=enhance_formula_text(text),
                section_path=section_path or f"Page {page_number}",
                page=page_number,
            )
        ]

    def parse(self, path: Path) -> List[CanonicalBlock]:
        try:
            import fitz
        except ImportError as e:
            raise RuntimeError(
                "解析 PDF 需要 PyMuPDF。\n"
                "请执行：pip install PyMuPDF"
            ) from e

        doc = fitz.open(path)

        try:
            body_size, heading_sizes = self._scan_font_statistics(doc)

            heading_stack: List[str] = []
            blocks: List[CanonicalBlock] = []

            for page_idx, page in enumerate(doc, start=1):
                tables = self._extract_tables(page)
                table_bboxes = [bbox for bbox, _ in tables]

                try:
                    data = page.get_text("dict", sort=True)
                except TypeError:
                    data = page.get_text("dict")

                events = []
                text_char_count = 0

                # 表格作为布局事件参与排序
                for bbox, md in tables:
                    events.append(
                        {
                            "kind": "table",
                            "bbox": bbox,
                            "content": md,
                        }
                    )

                for block in data.get("blocks", []):
                    if block.get("type") != 0:
                        continue

                    for line in block.get("lines", []):
                        spans = line.get("spans", [])

                        parts = []
                        sizes = []
                        bold = False

                        for span in spans:
                            text = str(span.get("text", ""))
                            if text:
                                parts.append(text)

                            try:
                                sizes.append(float(span.get("size", 0.0)))
                            except Exception:
                                pass

                            bold = bold or _is_bold(span)

                        line_text = "".join(parts).strip()
                        if not line_text:
                            continue

                        bbox = tuple(float(x) for x in line.get("bbox", (0, 0, 0, 0)))

                        # 已经识别为表格的区域，其文本层通常会重复出现，跳过。
                        if any(_bbox_overlap_ratio(bbox, tb) >= 0.55 for tb in table_bboxes):
                            continue

                        text_char_count += len(line_text)

                        max_size = max(sizes) if sizes else body_size
                        heading_level = self._heading_level(
                            line_text=line_text,
                            max_size=max_size,
                            is_bold=bold,
                            body_size=body_size,
                            heading_sizes=heading_sizes,
                        )

                        events.append(
                            {
                                "kind": "heading" if heading_level else "text",
                                "level": heading_level,
                                "bbox": bbox,
                                "content": line_text,
                            }
                        )

                # 扫描页 / 文本层严重不足 -> OCR fallback。
                if text_char_count < settings.ingestion.pdf.ocr_min_text_chars:
                    ocr_blocks = self._ocr_page(
                        page=page,
                        page_number=page_idx,
                        section_path=" > ".join(heading_stack),
                    )

                    if ocr_blocks:
                        # 进入 OCR fallback 时不丢弃已识别的表格：文本层稀疏不代表
                        # 没有表格，find_tables 可能已识别出真实表格，先落表再落
                        # OCR 文本，避免 continue 把表格事件一起丢掉。
                        for ev in events:
                            if ev.get("kind") == "table":
                                blocks.append(
                                    CanonicalBlock(
                                        content_type="table",
                                        content=ev["content"],
                                        section_path=" > ".join(heading_stack) or f"Page {page_idx}",
                                        page=page_idx,
                                    )
                                )
                        blocks.extend(ocr_blocks)
                        continue

                # y 优先、x 次优。
                events.sort(
                    key=lambda item: (
                        item["bbox"][1],
                        item["bbox"][0],
                    )
                )

                text_buf: List[str] = []
                text_buf_section = ""

                def flush_text():
                    nonlocal text_buf, text_buf_section
                    text = "\n".join(text_buf).strip()

                    if text:
                        blocks.append(
                            CanonicalBlock(
                                content_type="text",
                                content=enhance_formula_text(text),
                                section_path=text_buf_section or f"Page {page_idx}",
                                page=page_idx,
                            )
                        )

                    text_buf = []
                    text_buf_section = ""

                for event in events:
                    if event["kind"] == "heading":
                        flush_text()

                        level = int(event["level"] or 1)
                        title = event["content"].strip()

                        heading_stack = heading_stack[: level - 1]
                        heading_stack.append(title)
                        continue

                    if event["kind"] == "table":
                        flush_text()

                        blocks.append(
                            CanonicalBlock(
                                content_type="table",
                                content=event["content"],
                                section_path=" > ".join(heading_stack) or f"Page {page_idx}",
                                page=page_idx,
                            )
                        )
                        continue

                    section = " > ".join(heading_stack) or f"Page {page_idx}"

                    if text_buf and section != text_buf_section:
                        flush_text()

                    text_buf_section = section
                    text_buf.append(event["content"])

                flush_text()

            return blocks

        finally:
            doc.close()
