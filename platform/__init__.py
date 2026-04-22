"""MSG-Embedding platform namespace package (backend + worker + frontend).

Because this package shadows the stdlib ``platform`` module, we delegate
attribute access that is not defined here to the real stdlib module.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any

__all__ = ["backend", "worker"]


def _extend_path_for_tests() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    test_dir = os.path.join(repo_root, "tests", "unit", "platform")
    if os.path.isdir(test_dir) and test_dir not in __path__:
        __path__.append(test_dir)  # type: ignore[name-defined]


_extend_path_for_tests()


def _load_stdlib_platform():
    stdlib_dir = os.path.dirname(os.__file__)
    candidate = os.path.join(stdlib_dir, "platform.py")
    if not os.path.exists(candidate):
        return None
    spec = importlib.util.spec_from_file_location("_stdlib_platform", candidate)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_stdlib_platform", module)
    spec.loader.exec_module(module)
    return module


_stdlib_platform = _load_stdlib_platform()


def __getattr__(name: str) -> Any:
    if _stdlib_platform is not None and hasattr(_stdlib_platform, name):
        return getattr(_stdlib_platform, name)
    raise AttributeError(f"module 'platform' has no attribute {name!r}")
