# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import List

from utils.schema import CanonicalBlock
from parsers.base import BaseParser
from parsers.ocr_utils import ocr_text


class ImageParser(BaseParser):
    extensions = (
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    )

    document_type = "image"

    def parse(self, path: Path) -> List[CanonicalBlock]:
        text = ocr_text(path)

        if not text:
            return []

        return [
            CanonicalBlock(
                content_type="image_ocr",
                content=text,
                section_path=path.stem,
                image_index=1,
            )
        ]
