# -*- coding: utf-8 -*-
"""
跨格式公式增强模块

目标：对 PDF/DOCX/PPTX/XLSX/Markdown/TXT 抽取文本做无损规范化，
在不改变原文语义的前提下提升 Embedding 与 LLM 理解。

重要：本模块不做任何“公式恢复/补全”。PDF 抽取断裂的公式无法从文本可靠还原。
早期版本曾在宽松正则命中时追加一段硬编码工程公式
（如 ``L = VOUT*(VIN-VOUT)/(ΔIL*fS*VIN)``），会把猜测内容注入知识库，
是确定的幻觉源。现已移除，只保留原文、不做任何非原文注入。
"""

from __future__ import annotations


def normalize_pdf_formula(text: str) -> str:
    """对 PDF 抽取文本做无损处理。

    不重组、不补全断裂公式：断裂片段无法可靠还原，任何猜测式替换都会污染知识库，
    违背“喂给 LLM 的内容必须 KB-based 且正确”的验收目标。故原样返回。
    """
    return text


def enhance_formula_text(text: str) -> str:
    """公式文本增强：保留原文，不注入任何非原文内容。"""
    if not text:
        return text
    return normalize_pdf_formula(text)
