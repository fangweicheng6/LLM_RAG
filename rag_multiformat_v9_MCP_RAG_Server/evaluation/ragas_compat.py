"""Small compatibility helpers for Ragas prompts on OpenAI-compatible gateways.

Ragas 0.4.3 repairs an invalid metric output with ``fix_output_format``.  That
repair prompt itself expects ``StringIO`` (``{"text": "..."}``), while some
models return the repaired target JSON directly.  Preserve the repaired text
but add only the envelope Ragas expects; the original Ragas parser still owns
validation of the target schema afterwards.
"""
from __future__ import annotations

import json


RAGAS_FIX_OUTPUT_MARKER = (
    "The output string did not satisfy the constraints given in the prompt. "
    "Fix the output string and return it."
)


def normalize_ragas_fix_output(prompt_text: str, answer: str) -> str:
    """Adapt direct repaired JSON to Ragas 0.4.3's ``StringIO`` envelope.

    The transformation is deliberately limited to the Ragas repair prompt.
    Already-correct ``{"text": ...}`` responses are returned unchanged.
    """
    if RAGAS_FIX_OUTPUT_MARKER not in prompt_text:
        return answer

    stripped = answer.strip()
    try:
        parsed = json.loads(stripped)
    except (TypeError, ValueError):
        parsed = None

    if isinstance(parsed, dict) and isinstance(parsed.get("text"), str):
        inner = parsed.get("text", "").strip()
        # 兼容部分 OpenAI-compatible 网关将 JSON 再包进 text 字段。
        # Ragas 0.4.3 的 StringIO 修复格式需要 text，但 text 内部不应再嵌套同名 envelope。
        try:
            json.loads(inner)
            return json.dumps({"text": inner}, ensure_ascii=False)
        except Exception:
            return answer

    return json.dumps({"text": answer}, ensure_ascii=False)
