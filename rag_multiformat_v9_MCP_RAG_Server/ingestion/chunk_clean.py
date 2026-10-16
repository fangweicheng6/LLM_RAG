# -*- coding: utf-8 -*-
"""切片清洗与过滤（Docling 路径与行式解析器共用）。

职责：
1. 剥离解析器残留的 HTML 注释——Docling 对无法解码的公式会输出
   ``<!-- formula-not-decoded -->`` 占位符，必须移除；
2. 文本规范化（BOM / Unicode NFC / CRLF / 制表符 / 行尾空白 / 连续空行）；
3. 垃圾块判定：页码、公式编号、孤立数值、页眉页脚、URL/邮箱/Markdown 链接页脚；
4. 块级指纹，用于跨块 / 跨文件去重（与旧 ``content_fingerprint`` 口径一致）。
"""

from __future__ import annotations

import re
import unicodedata

# 不用 re.S：否则一个未闭合的 ``<!--`` 会跨行吞掉直到下一个 ``-->`` 之间的真实正文。
# 限制为单行匹配，宁可漏删一个多行注释，也不误删正文。
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->")
_TRAILING_SPACES_RE = re.compile(r"[ \t]+$", re.M)
_EXCESS_BLANK_RE = re.compile(r"\n{3,}")

# 单行页眉/页脚/Logo 噪声白名单（精确全大写匹配）。
# 说明：早期用“单行全大写 token”启发式（_ALLCAPS_TOKEN_RE.fullmatch）删除页眉，
# 但器件型号（LM2596 / UC3842 / MOSFET / IGBT 等）与应用笔记编号（AN90059）同样
# 形如全大写 token，是电力电子知识库的核心检索 token，误删会直接漏检。
# 故改为精确白名单，宁留勿删——宁可让个别页眉漏进正文，也不删掉可检索内容。
_HEADER_NOISE_TOKENS = frozenset({
    "INSTRUMENTS", "TEXAS", "RENESAS", "MPS", "TI", "ST", "NXP",
    "TOSHIBA", "INFINEON", "DATASHEET", "CONFIDENTIAL", "PROPRIETARY",
})
_URL_RE = re.compile(r"(?:https?://|www\.)[\w./?=&%\-+#@]*", re.I)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
_MD_LINK_RE = re.compile(r"\[[^\]\n]*\]\([^)\n]*\)")
_PAGE_FOOTER_RE = re.compile(r"Page \d+(?:\s+of\s+\d+)?", re.I)


def strip_html_comments(text: str) -> str:
    return _HTML_COMMENT_RE.sub("", text)


def normalize_text(text: str) -> str:
    text = text.replace("﻿", "")
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", "    ")
    text = _TRAILING_SPACES_RE.sub("", text)
    text = _EXCESS_BLANK_RE.sub("\n\n", text)
    return text.strip()


def content_fingerprint(text: str) -> str:
    """跨块 / 跨文件去重用的稳定指纹，不改变展示内容。"""
    return re.sub(r"\s+", " ", normalize_text(text)).strip().lower()


def clean_chunk_text(text: str) -> str:
    """切片正文的最终清洗：去公式占位符 + 规范化。"""
    return normalize_text(strip_html_comments(text))


def is_junk_chunk(text: str) -> bool:
    """判定清洗后的正文是否属于无检索价值的垃圾块。"""
    if not text:
        return True

    # 无任何字母/CJK/希腊字母/单位符号：页码、公式编号、孤立数值。
    # 电力电子正文常出现纯希腊字母或单位（Δ、Ω、μ、√、∑），只认 A-Za-z 会误删。
    if not re.search(r"[A-Za-z一-鿿µͰ-Ͽ℀-⅏∀-⋿]", text):
        return True

    t = text.strip()

    # URL / 邮箱 / 纯 Markdown 链接页脚。
    if _URL_RE.fullmatch(t):
        return True
    if _EMAIL_RE.fullmatch(t):
        return True
    if _MD_LINK_RE.fullmatch(t):
        return True

    # "Page N" / "Page N of M" 页脚。
    if _PAGE_FOOTER_RE.fullmatch(t):
        return True

    # 单行、且为已知页眉/Logo 白名单 token。
    if "\n" not in t and t.strip().upper() in _HEADER_NOISE_TOKENS:
        return True

    return False
