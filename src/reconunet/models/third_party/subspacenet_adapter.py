"""Adapter around the upstream SubspaceNet model (Shmuel et al., 2023).

Upstream repo (pinned via git submodule):
    third_party/subspacenet/  ↦  https://github.com/ShlezingerLab/SubspaceNet.git

The upstream model lives at ``src/models.py::SubspaceNet`` within the
submodule and depends on a flat ``src`` package laid out at the submodule's
root — so we ``sys.path.insert`` the submodule root before importing.  No
upstream code is modified.

Input/output contract
---------------------
* Input:  :math:`R^{\\tau}_x \\in \\mathbb{R}^{B \\times \\tau \\times 2M \\times M}`
  where the penultimate axis stacks ``real`` on top of ``imag`` so the model
  can run on real-valued convs.  This is produced by
  :func:`reconunet.data.scene_renderer.lag_stack`.
* Output tuple (upstream): ``(doa_prediction, doa_all_predictions, roots, Rz)``.
  We wrap ``doa_prediction`` (already in radians) into
  :class:`BaselineOutput.angles_pred` and stash the rest in ``extras``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ._base import BaselineAdapter, BaselineOutput


def _submodule_root() -> Path:
    """``<ReconUNet>/third_party/subspacenet`` as an absolute path."""
    # src/reconunet/models/third_party/subspacenet_adapter.py
    #   → ReconUNet = parents[4]
    return Path(__file__).resolve().parents[4] / "third_party" / "subspacenet"


def _ensure_on_path() -> None:
    """Insert the submodule root so ``from src.models import SubspaceNet`` works.

    The upstream repo uses a flat ``src/`` package (no ``__init__``-style
    shadowing) — because our editable install also registers a ``reconunet``
    package (not ``src``), the two do not collide.  Still, we push the
    submodule path *after* our own so ``reconunet.*`` always wins on name
    collisions.
    """
    root = _submodule_root()
    if not root.exists():  # pragma: no cover — guards a missing checkout
        raise FileNotFoundError(
            f"SubspaceNet submodule not found at {root}. "
            "Run `git submodule update --init third_party/subspacenet`."
        )
    path_str = str(root)
    if path_str not in sys.path:
        sys.path.append(path_str)


class SubspaceNetAdapter(BaselineAdapter):
    """Canonical-interface wrapper around ``subspacenet.src.models.SubspaceNet``."""

    name = "subspacenet"

    def __init__(self, tau: int = 8, M: int = 3, diff_method: str = "root_music") -> None:
        """Defaults match the paper's baseline (τ=8 lags, M=3 sources, Root-MUSIC head)."""
        self.tau = tau
        self.M = M
        self.diff_method = diff_method

    # --- construction -------------------------------------------------------

    def build_model(self, cfg: Dict[str, Any]) -> nn.Module:
        _ensure_on_path()
        # Upstream imports are sensitive to module name; defer to call-time.
        from src.models import SubspaceNet  # type: ignore

        tau = int(cfg.get("tau", self.tau))
        M = int(cfg.get("M", self.M))
        diff_method = str(cfg.get("diff_method", self.diff_method))
        model = SubspaceNet(tau=tau, M=M, diff_method=diff_method)
        return model

    # --- data-plane ---------------------------------------------------------

    def prepare_input(self, snapshots: torch.Tensor, meta: Dict[str, Any]) -> torch.Tensor:
        """Build :math:`R^{\\tau}_x` from ``snapshots`` shape ``[B, M, T]``.

        Delegated to :func:`reconunet.data.scene_renderer.lag_stack` so all three
        models consume the identical autocorrelation tensor (fair comparison).
        """
        from reconunet.data.scene_renderer import lag_stack

        tau = int(meta.get("tau", self.tau))
        return lag_stack(snapshots, tau=tau)

    def forward(
        self,
        model: nn.Module,
        prepped: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> BaselineOutput:
        doa_pred, doa_all, roots, Rz = model(prepped)
        loss = None
        if targets is not None and self.training_loss_fn is not None:
            loss = self.training_loss_fn(doa_pred, targets)
        return BaselineOutput(
            angles_pred=doa_pred,
            loss=loss,
            extras={"doa_all": doa_all, "roots": roots, "Rz": Rz},
        )

    # --- training loss (pluggable) -----------------------------------------

    training_loss_fn = None  # set to a callable(pred, target) -> scalar
