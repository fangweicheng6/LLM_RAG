# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from config import (
    WHISPER_MODEL,
    WHISPER_LANGUAGE,
    WHISPER_BEAM_SIZE,
    WHISPER_VAD_FILTER,
)


_WHISPER = None


def _get_whisper_model():
    global _WHISPER

    if _WHISPER is not None:
        return _WHISPER

    try:
        import torch
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise RuntimeError(
            "需要解析音频/视频，但未安装 faster-whisper。\n"
            "请执行：pip install faster-whisper"
        ) from e

    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
    else:
        device = "cpu"
        compute_type = "int8"

    _WHISPER = WhisperModel(
        WHISPER_MODEL,
        device=device,
        compute_type=compute_type,
    )

    return _WHISPER


def transcribe_media(path: Path) -> List[Tuple[float, float, str]]:
    """
    faster-whisper 可以通过 PyAV 直接读取常见音频和视频容器。

    返回：
    [
        (start_sec, end_sec, text),
        ...
    ]
    """
    model = _get_whisper_model()

    segments, _ = model.transcribe(
        str(path),
        language=WHISPER_LANGUAGE,
        beam_size=WHISPER_BEAM_SIZE,
        vad_filter=WHISPER_VAD_FILTER,
    )

    result: List[Tuple[float, float, str]] = []

    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue

        result.append(
            (
                float(seg.start),
                float(seg.end),
                text,
            )
        )

    return result
