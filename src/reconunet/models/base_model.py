"""Minimal abstract base class for all Tri4Net models.

Some deep-learning modules expect a common parent ``BaseModel`` with a
config dictionary attribute.  This file recreates a lightweight version so
that imports like ``from ..base_model import BaseModel`` work even when the
original file is missing.
"""
from __future__ import annotations

import abc
from typing import Any, Dict, Optional
import torch.nn as nn

__all__ = ["BaseModel"]


class BaseModel(nn.Module, metaclass=abc.ABCMeta):
    """Base class providing a *config* attribute and abstract *forward*."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config: Dict[str, Any] = config or {}

    # ------------------------------------------------------------------
    @abc.abstractmethod
    def forward(self, *inputs, **kwargs):  # noqa: D401 – abstract
        """Sub-classes must implement their forward pass."""

    # Optional convenience ------------------------------------------------
    def estimate_doa(self, *args, **kwargs):  # noqa: D401 – optional
        raise NotImplementedError("Model does not implement DoA estimation helper.") 