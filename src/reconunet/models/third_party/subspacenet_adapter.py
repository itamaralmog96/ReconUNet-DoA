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
  We normalise ``doa_prediction`` to the repo-canonical convention
  (**radians, broadside zero, range ``(-pi/2, pi/2)``**) before wrapping it in
  :class:`BaselineOutput.angles_pred`; the rest goes in ``extras``.

  Convention note: the differentiable ``root_music`` head
  (:mod:`reconunet.models.deep_learning.subspace_models`) returns *degrees* in
  ``[0, 180]`` (array-axis / cosine convention, ``= 90 + broadside_deg``), so it
  needs ``deg2rad(x - 90)`` — the exact conversion already applied by every other
  call site (``cli/evaluate_scenarios.py``, ``cli/train.py``, ``cli/evaluate.py``).
  The ``esprit`` head already returns broadside radians and passes through.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
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


def _pad_to(x: torch.Tensor, width: int) -> torch.Tensor:
    """Right-pad ``[g, k]`` angles to ``[g, width]`` with NaN (NaN-padded slots
    mark "no source"; the RMSPE loss / val metric mask them out)."""
    g, k = x.shape
    if k >= width:
        return x[:, :width]
    pad = torch.full((g, width - k), float("nan"), device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=1)


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

        # Bypass the model's internal *fixed-M* subspace head.  We re-derive the
        # angles from the surrogate covariance ``Rz`` in :meth:`forward` using the
        # **true per-sample source count** (so one model generalises across
        # ``k_choices = [1,2,3,4]``, exactly like the ReconUNet path).  Stubbing
        # the head also avoids running Root-MUSIC twice per forward.  The local
        # eigh-based ``root_music`` / ``esprit`` (torch.linalg.eigh, correct for
        # the Hermitian Rz = K^H K + εI) are applied per-source-count instead.
        model.diff_method = lambda Rz, M_, B_: (None, None, None)
        self.diff_method = diff_method
        return model

    # --- angle extraction --------------------------------------------------

    def _angles_from_Rz(self, Rz: torch.Tensor, k: int) -> torch.Tensor:
        """Estimate exactly ``k`` broadside-radian angles from ``Rz`` ``[g,N,N]``.

        ``root_music`` returns degrees in ``[0,180]`` → ``deg2rad(x-90)``;
        ``esprit`` already returns broadside radians.
        """
        from reconunet.models.deep_learning.subspace_models import esprit, root_music

        if str(self.diff_method).lower().startswith("root_music"):
            deg, _, _ = root_music(Rz, k, Rz.shape[0])
            return torch.deg2rad(deg[:, :k] - 90.0)
        return esprit(Rz, k, Rz.shape[0])[:, :k]

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
        _, _, _, Rz = model(prepped)                       # surrogate covariance [B,N,N]
        B = Rz.shape[0]

        # Per-sample source count: estimate exactly K_true angles per sample so a
        # single model generalises across k_choices=[1,2,3,4] (apples-to-apples
        # with ReconUNet).  Falls back to the fixed self.M when n_sources is not
        # threaded through (e.g. a fixed-K corpus or a caller that omits it).
        n_sources = (meta or {}).get("n_sources", None) if meta else None

        if n_sources is None:
            angles_pred = self._angles_from_Rz(Rz, int(self.M))
        else:
            ns = (n_sources.detach().cpu().numpy() if isinstance(n_sources, torch.Tensor)
                  else np.asarray(n_sources)).astype(int)
            K_max = max(int(ns.max()), 1)
            angles_pred = torch.full((B, K_max), float("nan"),
                                     device=Rz.device, dtype=torch.float32)
            for k in np.unique(ns):
                if k < 1:
                    continue
                mask = torch.from_numpy(ns == k).to(Rz.device)
                ang_k = self._angles_from_Rz(Rz[mask], int(k))     # [g, k]
                angles_pred = angles_pred.index_put(
                    (mask.nonzero(as_tuple=True)[0],), _pad_to(ang_k, K_max))

        loss = None
        if targets is not None and self.training_loss_fn is not None:
            loss = self.training_loss_fn(angles_pred, targets)
        return BaselineOutput(angles_pred=angles_pred, loss=loss, extras={"Rz": Rz})

    # --- training loss (pluggable) -----------------------------------------

    training_loss_fn = None  # set to a callable(pred, target) -> scalar
