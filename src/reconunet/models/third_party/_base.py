"""Abstract ``BaselineAdapter`` and shared ``BaselineOutput`` container.

All third-party baselines (SubspaceNet, DOA-ViT / SubViT) ultimately need the
same *observation*: a batch of complex snapshot matrices :math:`X \\in
\\mathbb{C}^{B \\times M \\times T}`.  What they expect at their ``forward``
boundary, however, differs:

* SubspaceNet consumes a **lag-stack autocorrelation tensor**
  :math:`R^{\\tau}_x \\in \\mathbb{R}^{B \\times \\tau \\times 2M \\times M}`
  (real/imag stacked along the penultimate axis).
* DOA-ViT consumes a **spatial covariance matrix**
  :math:`K_x \\in \\mathbb{C}^{B \\times M \\times M}`, typically presented as
  real/imag channels :math:`[B, 2, M, M]`.
* ReconUNet itself consumes the same lag-stack tensor as SubspaceNet.

This common interface lets the unified training / evaluation harness feed the
same ``(snapshots, angles_true)`` batch to every model without leaking
model-specific tensor-shape plumbing into calling code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class BaselineOutput:
    """Canonical output of a single ``BaselineAdapter.forward`` call.

    Attributes
    ----------
    angles_pred
        Predicted DoAs in **radians**, shape ``[B, K]`` (K = number of sources).
        Angles are in the open interval ``(-π/2, π/2)`` for a broadside ULA,
        or ``(-π, π)`` for a 2-D scan.  Trailing columns may be padded with
        NaN if K < max_K (e.g. when sources count varies across the batch).
    loss
        Optional training-time loss scalar, shape ``[]`` or ``[1]``.  ``None``
        at inference time.
    extras
        Free-form dictionary for model-specific auxiliary outputs:
        surrogate covariance matrices, root-polynomial roots, attention maps,
        etc.  The evaluation harness only inspects ``angles_pred``; everything
        else is for logging or downstream experimentation.
    """

    angles_pred: torch.Tensor
    loss: Optional[torch.Tensor] = None
    extras: Dict[str, Any] = field(default_factory=dict)


class BaselineAdapter(ABC):
    """Abstract interface every baseline adapter implements.

    Sub-classes are **stateless plumbing** — they do not own the ``nn.Module``
    instance.  They build it on request (``build_model``), transform the shared
    snapshot batch into the specific tensor the model expects
    (``prepare_input``), and interpret the raw output tuple into a canonical
    :class:`BaselineOutput` (``forward``).

    The adapter keeps a reference to the caller's source-of-truth
    ``SceneRenderer`` so it can regenerate autocorrelation lag stacks,
    covariance matrices, or whatever else it needs deterministically.
    """

    name: str = "baseline"
    requires: tuple = ()  # e.g. ("scipy", "einops")

    # --- construction / lifecycle ------------------------------------------

    @abstractmethod
    def build_model(self, cfg: Dict[str, Any]) -> nn.Module:
        """Instantiate the upstream model from a config dictionary."""

    def load_checkpoint(self, model: nn.Module, path: str, strict: bool = True) -> None:
        """Default ``torch.load`` + ``load_state_dict``.  Override if the
        upstream checkpoint format is a dict of dicts or contains an optimizer
        state we want to drop."""
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=strict)

    # --- data-plane --------------------------------------------------------

    @abstractmethod
    def prepare_input(self, snapshots: torch.Tensor, meta: Dict[str, Any]) -> torch.Tensor:
        """Convert ``snapshots`` of shape ``[B, M, T]`` (complex) into the
        exact tensor the model consumes.  ``meta`` carries the shared array
        config (``M``, ``tau``, ``fs``, etc.)."""

    @abstractmethod
    def forward(
        self,
        model: nn.Module,
        prepped: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> BaselineOutput:
        """Run ``model`` on ``prepped`` and normalize its output."""

    # --- optimizer / scheduler helpers ------------------------------------

    def default_optimizer(
        self, model: nn.Module, lr: float = 1e-3, weight_decay: float = 0.0
    ) -> torch.optim.Optimizer:
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --- convenience --------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover — cosmetic
        return f"<{self.__class__.__name__} name={self.name!r}>"
