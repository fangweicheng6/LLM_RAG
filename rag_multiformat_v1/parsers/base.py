# -*- coding: utf-8 -*-
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from schema import CanonicalBlock


class BaseParser(ABC):
    """
    所有 Parser 的统一接口。

    新增格式只需新增一个 *_parser.py：
        class XxxParser(BaseParser):
            extensions = (".xxx",)
            document_type = "xxx"

            def parse(self, path):
                return [CanonicalBlock(...), ...]

    parsers/__init__.py 会自动发现并注册，无需手工维护映射表。
    """

    extensions: tuple[str, ...] = ()
    document_type: str = "document"

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    @abstractmethod
    def parse(self, path: Path) -> List[CanonicalBlock]:
        raise NotImplementedError
