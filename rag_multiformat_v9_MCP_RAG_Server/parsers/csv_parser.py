# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from typing import List

from utils.schema import CanonicalBlock
from parsers.base import BaseParser
from parsers.formula_utils import enhance_formula_text
from parsers.common import row_to_retrieval_text, read_text_safely


class CSVParser(BaseParser):
    extensions = (".csv",)
    document_type = "spreadsheet"

    def parse(self, path: Path) -> List[CanonicalBlock]:
        raw = read_text_safely(path)
        # 不能用 raw.splitlines()：它会连同「引号字段内的换行」一起切分，
        # 破坏带内嵌换行的合法 CSV 行。交给 csv.reader 直接读 StringIO，
        # 由 csv 模块正确处理引号内的换行。
        reader = csv.reader(StringIO(raw))
        rows = list(reader)

        if not rows:
            return []

        headers = rows[0]
        blocks: List[CanonicalBlock] = []

        for row_idx, row in enumerate(rows[1:], start=2):
            if not any(str(v).strip() for v in row):
                continue

            content = row_to_retrieval_text(headers, row)
            if not content:
                continue

            blocks.append(
                CanonicalBlock(
                    content_type="table_row",
                    content=enhance_formula_text(content),
                    section_path=path.stem,
                    sheet_name=path.stem,
                    row_start=row_idx,
                    row_end=row_idx,
                )
            )

        return blocks
