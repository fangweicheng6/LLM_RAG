# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import List

from utils.schema import CanonicalBlock
from parsers.base import BaseParser
from parsers.formula_utils import enhance_formula_text
from parsers.common import read_text_safely


class TxtParser(BaseParser):
    extensions = (".txt",)
    document_type = "text"

    def parse(self, path: Path) -> List[CanonicalBlock]:
        text = read_text_safely(path)

        return [
            CanonicalBlock(
                content_type="text",
                content=enhance_formula_text(text),
                section_path=path.stem,
            )
        ]
