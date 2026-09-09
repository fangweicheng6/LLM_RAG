# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Dict, List

from parsers.base import BaseParser


_PARSER_BY_EXTENSION: Dict[str, BaseParser] | None = None
_PARSERS: List[BaseParser] | None = None


def _discover_parsers():
    """
    自动扫描 parsers 包中的 *_parser.py。
    新增格式后只需要新增文件，不需要修改注册表。
    """
    global _PARSER_BY_EXTENSION, _PARSERS

    if _PARSER_BY_EXTENSION is not None:
        return

    parser_by_ext: Dict[str, BaseParser] = {}
    parsers: List[BaseParser] = []

    package_dir = Path(__file__).resolve().parent

    for module_info in pkgutil.iter_modules([str(package_dir)]):
        name = module_info.name

        if not name.endswith("_parser"):
            continue

        module = importlib.import_module(f"parsers.{name}")

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj is BaseParser:
                continue

            if not issubclass(obj, BaseParser):
                continue

            # 只实例化定义在当前模块中的 Parser，避免重复发现 import 进来的类。
            if obj.__module__ != module.__name__:
                continue

            parser = obj()

            if not parser.extensions:
                continue

            parsers.append(parser)

            for ext in parser.extensions:
                ext = ext.lower()

                if ext in parser_by_ext:
                    raise RuntimeError(
                        f"Parser 扩展名冲突：{ext}\n"
                        f"{parser_by_ext[ext].__class__.__name__} 与 "
                        f"{parser.__class__.__name__}"
                    )

                parser_by_ext[ext] = parser

    _PARSERS = parsers
    _PARSER_BY_EXTENSION = parser_by_ext


def get_parser(path: Path) -> BaseParser | None:
    _discover_parsers()
    return _PARSER_BY_EXTENSION.get(path.suffix.lower())


def supported_extensions() -> set[str]:
    _discover_parsers()
    return set(_PARSER_BY_EXTENSION.keys())


def all_parsers() -> List[BaseParser]:
    _discover_parsers()
    return list(_PARSERS)
