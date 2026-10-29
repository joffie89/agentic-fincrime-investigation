"""Synthetic, executable quality evaluation."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["run_evaluation"]


def __getattr__(name: str) -> Any:
    if name != "run_evaluation":
        raise AttributeError(name)
    value = import_module("app.evaluation.evaluate").run_evaluation
    globals()[name] = value
    return value
