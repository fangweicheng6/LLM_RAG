# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import List

from utils.schema import CanonicalBlock
from parsers.base import BaseParser
from parsers.asr_utils import transcribe_media


class AudioParser(BaseParser):
    extensions = (
        ".mp3",
        ".wav",
        ".m4a",
        ".flac",
        ".aac",
        ".ogg",
        ".wma",
        ".opus",
    )

    document_type = "audio"

    def parse(self, path: Path) -> List[CanonicalBlock]:
        segments = transcribe_media(path)

        return [
            CanonicalBlock(
                content_type="transcript",
                content=text,
                section_path="Transcript",
                media_start_sec=start,
                media_end_sec=end,
            )
            for start, end, text in segments
        ]
