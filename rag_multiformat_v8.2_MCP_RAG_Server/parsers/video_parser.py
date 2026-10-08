# -*- coding: utf-8 -*-
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

from utils.schema import CanonicalBlock
from parsers.base import BaseParser
from parsers.asr_utils import transcribe_media
from parsers.ocr_utils import ocr_text
from utils.config_loader import settings


class VideoParser(BaseParser):
    extensions = (
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
        ".m4v",
        ".wmv",
        ".flv",
    )

    document_type = "video"

    def _frame_ocr(self, path: Path) -> List[CanonicalBlock]:
        if not settings.ingestion.video.frame_ocr_enabled:
            return []

        try:
            import cv2
        except ImportError as e:
            raise RuntimeError(
                "视频抽帧 OCR 需要 opencv-python-headless。\n"
                "请执行：pip install opencv-python-headless"
            ) from e

        cap = cv2.VideoCapture(str(path))

        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0

        if fps <= 0 or frame_count <= 0:
            cap.release()
            return []

        duration = frame_count / fps
        t = 0.0

        blocks: List[CanonicalBlock] = []
        last_text = None

        try:
            while t <= duration:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                ok, frame = cap.read()

                if not ok:
                    t += settings.ingestion.video.frame_ocr_interval_sec
                    continue

                with tempfile.NamedTemporaryFile(
                    suffix=".jpg",
                    delete=False,
                ) as tmp:
                    tmp_path = Path(tmp.name)

                try:
                    cv2.imwrite(str(tmp_path), frame)
                    text = ocr_text(tmp_path).strip()
                finally:
                    tmp_path.unlink(missing_ok=True)

                if text:
                    if not (
                        settings.ingestion.video.frame_ocr_dedup
                        and last_text is not None
                        and text == last_text
                    ):
                        blocks.append(
                            CanonicalBlock(
                                content_type="video_frame_ocr",
                                content=text,
                                section_path="Frame OCR",
                                media_start_sec=t,
                                media_end_sec=t,
                            )
                        )
                        last_text = text

                t += settings.ingestion.video.frame_ocr_interval_sec

        finally:
            cap.release()

        return blocks

    def parse(self, path: Path) -> List[CanonicalBlock]:
        blocks: List[CanonicalBlock] = []

        # 1) 音轨 ASR
        for start, end, text in transcribe_media(path):
            blocks.append(
                CanonicalBlock(
                    content_type="transcript",
                    content=text,
                    section_path="Transcript",
                    media_start_sec=start,
                    media_end_sec=end,
                )
            )

        # 2) 视频画面抽帧 OCR
        blocks.extend(self._frame_ocr(path))

        # 按时间排序：同一时间先 transcript 再 frame OCR。
        blocks.sort(
            key=lambda b: (
                b.media_start_sec if b.media_start_sec is not None else 0.0,
                0 if b.content_type == "transcript" else 1,
            )
        )

        return blocks
