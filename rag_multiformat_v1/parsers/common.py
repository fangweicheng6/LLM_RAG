# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path


def read_text_safely(path: Path) -> str:
    encodings = ("utf-8-sig", "utf-8", "gb18030")
    last_error = None

    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError as e:
            last_error = e

    raise RuntimeError(f"无法解码文本文件：{path}，最后错误：{last_error}")


def row_to_retrieval_text(headers, values) -> str:
    """
    将表格一行转成更适合 RAG 的文本：
    电压: 48.2；电流: 5.01；温度: 25.3
    """
    pairs = []
    max_len = max(len(headers), len(values))

    for i in range(max_len):
        header = str(headers[i]).strip() if i < len(headers) and headers[i] is not None else f"col_{i+1}"
        value = values[i] if i < len(values) else None

        if value is None or str(value).strip() == "":
            continue

        pairs.append(f"{header}: {value}")

    return "；".join(pairs)
