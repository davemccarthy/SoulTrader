"""
Advisor modules — lazy imports so heavy optional deps (e.g. polygon → pandas_ta)
load only when that submodule is used. Importing fda alone stays lightweight.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_MODULES = frozenset(
    {
        "polygon",
        "story",
        "fda",
        "pharm",
        "bizfeed",
        "insider",
        "user",
        "intraday",
        "flux",
        "vunder",
        "oscilla",
        "edgar",
    }
)


class _AdvisorsBundle:
    """Namespace matching the old `advisors` object; attributes load submodules lazily."""

    def __getattr__(self, name: str) -> Any:
        if name not in _LAZY_MODULES:
            raise AttributeError(name)
        mod = importlib.import_module(f"core.services.advisors.{name}")
        object.__setattr__(self, name, mod)
        return mod


advisors = _AdvisorsBundle()


def __getattr__(name: str) -> Any:
    if name == "advisors":
        return advisors
    if name in _LAZY_MODULES:
        mod = importlib.import_module(f"core.services.advisors.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals().keys(), "advisors", *_LAZY_MODULES})
