"""Lightweight model registry for dynamic instantiation.

Original file was missing; this replacement offers the decorator-based API
already used throughout the code-base.
"""
from __future__ import annotations

from typing import Dict, Type

__all__ = ["ModelRegistry"]


class ModelRegistry:
    _registry: Dict[str, Type] = {}

    # ------------------------------------------------------------------
    @classmethod
    def register(cls, name: str):
        """Decorator: ``@ModelRegistry.register('my_model')``."""

        def decorator(model_cls):
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    # ------------------------------------------------------------------
    @classmethod
    def get(cls, name: str):
        return cls._registry[name]

    @classmethod
    def available(cls):
        return list(cls._registry.keys()) 