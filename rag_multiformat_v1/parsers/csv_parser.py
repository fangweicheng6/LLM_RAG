# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
from pathlib import Path
from typing import List

from schema import CanonicalBlock
from parsers.base import BaseParser
from parsers.common import row_to_retrieval_text, read_text_safely


class CSVParser(BaseParser):
    extensions = (".csv",)
    document_type = "spreadsheet"

    def parse(self, path: Path) -> List[CanonicalBlock]:
        raw = read_text_safely(path)
        reader = csv.reader(raw.splitlines())
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
                    content=content,
                    section_path=path.stem,
                    sheet_name=path.stem,
                    row_start=row_idx,
                    row_end=row_idx,
                )
            )

        return blocks
