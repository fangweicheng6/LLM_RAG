# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

from utils.schema import CanonicalBlock
from parsers.base import BaseParser
from parsers.formula_utils import enhance_formula_text
from parsers.common import row_to_retrieval_text

# 含字母/CJK 的单元格（宽松判定，用于“表头行”检测）。
_LETTER_RE = re.compile(r"[A-Za-z一-鿿]")


def _has_text(value: str) -> bool:
    return bool(_LETTER_RE.search(value))


def _is_label(value: str) -> bool:
    """描述性标签（严格判定，用于表单 label: value 配对）。

    与“数值/短单位/型号代码”区分：CJK 文本、含空格/冒号/逗号/括号的多词短语、
    或长度足够的纯英文视为标签；含数字的英文 token（EE16 / EPC2152 / REV_0.4）
    与极短 token（V / Vac / kHz / Ts）视为值或单位。
    """
    v = value.strip()
    if len(v) < 2:
        return False
    if not _LETTER_RE.search(v):
        return False
    if re.search(r"[一-鿿]", v):
        return True
    if re.search(r"[\s:,(]", v):
        return True
    if re.search(r"\d", v):
        return False
    return len(v) >= 5


def _gather_cells(ws) -> Dict[int, List[Tuple[int, str]]]:
    """按行收集非空单元格 (列, 值)。列从 1 开始，值已去首尾空白。"""
    rows: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    for row in ws.iter_rows():
        for cell in row:
            value = cell.value
            if value is None or not str(value).strip():
                continue
            rows[cell.row].append((cell.column, str(value).strip()))
    return {r: sorted(cells) for r, cells in rows.items()}


def _sub_header_row(rows, header_row, header_cols):
    """检测紧邻表头行下方的“子表头行”（两级表头，如 Vergiga 的 RDON 列）。

    子表头行的单元格都落在表头列范围内、都是“短数值+单位”的 token，
    且至少填充一个“表头为空”或“表头以 = 结尾”的列。返回行号；不是子表头返回 None。
    """
    sorted_rows = sorted(rows)
    idx = sorted_rows.index(header_row)
    if idx + 1 >= len(sorted_rows):
        return None
    cells = rows[sorted_rows[idx + 1]]
    if not cells:
        return None

    min_col = min(header_cols)
    max_col = max(header_cols)
    fills = 0
    for col, v in cells:
        if not (min_col <= col <= max_col):
            return None
        if not re.search(r"\d", v):
            return None
        if len(v) > 12:
            return None
        if col not in header_cols or str(header_cols[col]).rstrip().endswith("="):
            fills += 1
    return sorted_rows[idx + 1] if fills > 0 else None


def _merge_sub_header(header_cols, sub_cells):
    """把子表头行的值并入表头：空表头列继承左侧最近表头文本，已有表头列直接拼接。"""
    merged = dict(header_cols)
    original = dict(header_cols)
    min_col = min(header_cols)
    for col, v in sorted(sub_cells):
        if col in original:
            merged[col] = str(original[col]).rstrip() + v
        else:
            base = None
            for c in range(col - 1, min_col - 1, -1):
                if c in original:
                    base = str(original[c]).rstrip()
                    break
            merged[col] = (base if base is not None else f"col_{col}") + v
    return merged


def _detect_table(rows: Dict[int, List[Tuple[int, str]]]):
    """检测“表头 + 对齐数据”结构。

    1. 表头行 = 含字母/CJK 单元格最多的行（宽表头优先于标题行/单位行）；
    2. 表头列下方的数据行，其单元格须以“数值（无字母/CJK）”为主，
       否则判定为表单（表单每一行的 label 列会重复出现，数据行仍全是文本）。

    返回 (header_row, {列: 表头文本}, 子表头行号)；非表格（表单）返回 (None, {}, None)。
    子表头行号在无两级表头时为 None，供 _table_rows 跳过该行。
    """
    best_row = None
    best_cols: Dict[int, str] = {}
    best_score = 0

    for r in sorted(rows):
        text_cells = [(col, v) for col, v in rows[r] if _has_text(v)]
        if len(text_cells) > best_score:
            best_score = len(text_cells)
            best_row = r
            best_cols = {col: v for col, v in text_cells}

    if best_score < 2 or best_row is None:
        return None, {}, None

    # 表头列 = 表头行里含字母/CJK 的列。
    header_cols = set(best_cols)

    # 表格：数据行主要填满表头列（on_grid），表头列之外几乎不落数据；
    # 表单：值/单位落在表头列之外的相邻列（off_grid 占比高）。
    on_grid = off_grid = 0
    for r in sorted(rows):
        if r <= best_row:
            continue
        for col, v in rows[r]:
            if col in header_cols:
                on_grid += 1
            else:
                off_grid += 1

    if on_grid >= 2 and off_grid <= max(1, on_grid * 0.2):
        sub_row = _sub_header_row(rows, best_row, best_cols)
        if sub_row is not None:
            best_cols = _merge_sub_header(best_cols, rows[sub_row])
        return best_row, best_cols, sub_row

    return None, {}, None


class XLSXParser(BaseParser):
    extensions = (".xlsx", ".xlsm")
    document_type = "spreadsheet"

    def parse(self, path: Path) -> List[CanonicalBlock]:
        try:
            from openpyxl import load_workbook
        except ImportError as e:
            raise RuntimeError("解析 Excel 需要 openpyxl。\n请执行：pip install openpyxl") from e

        wb = load_workbook(path, read_only=True, data_only=True)
        blocks: List[CanonicalBlock] = []

        try:
            for ws in wb.worksheets:
                rows = _gather_cells(ws)
                if not rows:
                    continue

                header_row, header_cols, sub_row = _detect_table(rows)
                if header_row is not None:
                    blocks.extend(self._table_rows(ws, rows, header_row, header_cols, sub_row))
                else:
                    blocks.extend(self._form_pairs(ws, rows))
        finally:
            wb.close()

        return blocks

    def _table_rows(self, ws, rows, header_row, header_cols, sub_row=None) -> List[CanonicalBlock]:
        cols = sorted(header_cols)
        headers = [header_cols[c] for c in cols]
        blocks: List[CanonicalBlock] = []

        for r in sorted(rows):
            if r <= header_row:
                continue
            if r == sub_row:
                continue
            row_map = dict(rows[r])
            values = [row_map.get(c) for c in cols]
            non_empty = [v for v in values if v is not None]
            if not non_empty:
                continue

            # 跳过「单位/注释行」：表头列下多数单元格是带括号的单位/注释
            # （如 "( cm4 )"、"@100℃(W)"），而非真实数据值。
            # 不能用「是否含字母」判断——"10 uH" 这类带单位的值也含字母，
            # 会误删合法数据行（回归：guide.xlsx 的 Inductor 行曾被整行跳过）。
            paren = sum(1 for v in non_empty if "(" in v)
            if len(non_empty) >= 2 and paren * 2 >= len(non_empty):
                continue

            # 分组标题行：整行只有一个描述性标签单元格（如 "TYPE EC CORE"、
            # "Step 1: Operational Specs"），是合并单元格的分节标题而非数据行。
            if len(non_empty) == 1 and _is_label(non_empty[0]):
                continue

            content = row_to_retrieval_text(headers, values)
            if not content:
                continue
            blocks.append(
                CanonicalBlock(
                    content_type="table_row",
                    content=enhance_formula_text(content),
                    section_path=ws.title,
                    sheet_name=ws.title,
                    row_start=r,
                    row_end=r,
                )
            )
        return blocks

    def _form_pairs(self, ws, rows) -> List[CanonicalBlock]:
        """表单型 sheet：把「标签 + 右侧相邻值/单位」配成 key: value。"""
        blocks: List[CanonicalBlock] = []

        for r in sorted(rows):
            row_cells = rows[r]
            for i, (col, label) in enumerate(row_cells):
                if not _is_label(label):
                    continue
                rights: List[str] = []
                for _, v in row_cells[i + 1:]:
                    if _is_label(v):
                        break  # 遇到下一个标签，结束本组
                    rights.append(v)
                if not rights:
                    continue
                label = label.rstrip(":：").strip()
                content = f"{label}: {' '.join(rights)}"
                blocks.append(
                    CanonicalBlock(
                        content_type="table_row",
                        content=enhance_formula_text(content),
                        section_path=ws.title,
                        sheet_name=ws.title,
                        row_start=r,
                        row_end=r,
                    )
                )
        return blocks
