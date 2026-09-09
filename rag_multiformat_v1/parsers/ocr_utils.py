# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from config import OCR_ENABLED, OCR_MIN_SCORE


_OCR_ENGINE = None


def _get_engine():
    global _OCR_ENGINE

    if not OCR_ENABLED:
        return None

    if _OCR_ENGINE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as e:
            raise RuntimeError(
                "需要 OCR，但未安装 rapidocr_onnxruntime。\n"
                "请执行：pip install rapidocr_onnxruntime"
            ) from e

        _OCR_ENGINE = RapidOCR()

    return _OCR_ENGINE


def ocr_image(path: Path) -> List[Tuple[str, float]]:
    """
    返回：
    [
        ("识别文本", 0.98),
        ...
    ]
    """
    engine = _get_engine()
    if engine is None:
        return []

    result, _ = engine(str(path))

    items: List[Tuple[str, float]] = []

    for item in result or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue

        text = str(item[1]).strip()
        if not text:
            continue

        try:
            score = float(item[2]) if len(item) >= 3 else 1.0
        except Exception:
            score = 1.0

        if score >= OCR_MIN_SCORE:
            items.append((text, score))

    return items


def ocr_text(path: Path) -> str:
    return "\n".join(text for text, _ in ocr_image(path)).strip()
