"""Single source of truth for runtime configuration.

Every setting is loaded from ``configs/*.yaml``.  The loader deliberately
exposes one structured object (``settings``); the former flat constants from
``config.py`` are gone, so changing YAML changes the running service.
"""
from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

try:
    import yaml
except ImportError:  # helpful for lightweight tooling; production uses PyYAML
    yaml = None


_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _ROOT / "configs"
_PATH_KEYS = {
    "kb_root", "data_dir", "processed_dir", "chunk_jsonl", "error_jsonl",
    "chroma_dir", "source_store_dir", "dir", "dataset", "benchmark_dir",
    "result_dir",
}


def _read_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        value = yaml.safe_load(text)
    else:
        # The checked-in files are JSON-compatible YAML.  This keeps config
        # inspection commands usable before dependencies are installed.
        value = json.loads("\n".join(line for line in text.splitlines()
                                      if not line.lstrip().startswith("#")))
    return value if isinstance(value, dict) else {}


def _merge(left: dict[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in right.items():
        if isinstance(value, Mapping) and isinstance(left.get(key), dict):
            _merge(left[key], value)
        else:
            left[key] = copy.deepcopy(value)
    return left


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _resolve_env(value: Any) -> Any:
    """Interpolate ``${VAR}`` / ``${VAR:-default}`` placeholders from os.environ.

    Unresolved placeholders (no env var and no default) are left as-is so a
    misconfiguration is visible instead of silently becoming an empty string.
    """
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    if isinstance(value, str):
        def _sub(match: re.Match) -> str:
            name = match.group(1)
            default = match.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            return match.group(0)
        return _ENV_PATTERN.sub(_sub, value)
    return value


def _resolve_paths(value: Any, *, parent_key: str = "") -> Any:
    if isinstance(value, dict):
        return {k: _resolve_paths(v, parent_key=k) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_paths(v, parent_key=parent_key) for v in value]
    if parent_key in _PATH_KEYS and isinstance(value, str):
        path = Path(value).expanduser()
        return path if path.is_absolute() else (_ROOT / path).resolve()
    return value


class ConfigNode:
    """Read-only nested mapping with attribute and dotted-path access."""

    def __init__(self, data: Mapping[str, Any]):
        object.__setattr__(self, "_data", {
            key: ConfigNode(value) if isinstance(value, Mapping)
            else tuple(ConfigNode(v) if isinstance(v, Mapping) else v for v in value)
            if isinstance(value, list) else value
            for key, value in data.items()
        })
        object.__setattr__(self, "_defaults", dict(self._data))

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_data":
            object.__setattr__(self, name, value)
        elif name in self._data:
            self._data[name] = value
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        # unittest.mock.patch restores attributes by calling delattr when the
        # original value came from __getattr__. Restore the YAML value instead
        # of deleting the structured setting permanently.
        if name in self._defaults:
            self._data[name] = self._defaults[name]
            return
        object.__delattr__(self, name)

    def get(self, path: str, default: Any = None) -> Any:
        current: Any = self
        for part in path.split("."):
            if isinstance(current, ConfigNode):
                current = current._data.get(part, default)
            elif isinstance(current, dict):
                current = current.get(part, default)
            else:
                return default
            if current is default:
                return default
        return current

    def as_dict(self) -> dict[str, Any]:
        def unwrap(value: Any) -> Any:
            if isinstance(value, ConfigNode):
                return {k: unwrap(v) for k, v in value._data.items()}
            if isinstance(value, (list, tuple)):
                return [unwrap(v) for v in value]
            return value
        return unwrap(self)


def load_settings() -> ConfigNode:
    merged: dict[str, Any] = {}
    owners: dict[str, str] = {}

    def leaves(value: Any, prefix: str = ""):
        if isinstance(value, Mapping):
            for key, child in value.items():
                yield from leaves(child, f"{prefix}.{key}" if prefix else key)
        else:
            yield prefix

    for path in sorted(_CONFIG_DIR.glob("*.yaml")):
        document = _read_yaml(path)
        for key in leaves(document):
            if key in owners:
                raise ValueError(
                    f"配置项 {key!r} 同时出现在 {owners[key]} 和 {path.name}；"
                    "请保留一个 YAML 定义，避免配置来源不明确。"
                )
            owners[key] = path.name
        _merge(merged, document)
    return ConfigNode(_resolve_paths(_resolve_env(merged)))


settings = load_settings()


def reload_settings() -> ConfigNode:
    """Reload YAML after an operator edits it in a long-lived process.

    关键：所有 ``from utils.config_loader import settings`` 的导入方持有的是
    ``settings`` 模块全局对象的引用。若只重新绑定模块全局（settings = ...），
    那些已导入的旧引用仍指向旧值，reload 形同虚设。因此这里原地替换现有
    ConfigNode 的 _data/_defaults，让所有导入方都能看到新配置。
    """
    global settings
    fresh = load_settings()
    settings._data = fresh._data
    settings._defaults = fresh._defaults
    return settings

