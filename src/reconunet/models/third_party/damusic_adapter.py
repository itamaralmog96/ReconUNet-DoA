"""Adapter around the upstream DA-MUSIC model (Merkofer et al., ICASSP 2022).

Upstream code (pinned via git submodule, **not modified**):
    third_party/subspacenet/src/models.py::DeepAugmentedMUSIC
    ↦ https://github.com/ShlezingerLab/SubspaceNet.git

DA-MUSIC ("Deep Augmented MUSIC") is the learned-front-end baseline named by
all three *Sensors* reviewers next to SubspaceNet.  A GRU runs over the raw
snapshots, a linear layer maps its final state to a learned N×N surrogate
covariance, MUSIC's noise-subspace spectrum is evaluated on a 361-point grid,
and an MLP regresses the ``M`` DoAs from that spectrum.

Input/output contract
---------------------
* Input:  complex snapshots :math:`X \\in \\mathbb{C}^{B \\times N \\times T}` —
  produced by :class:`reconunet.data.scene_dataset.DAMUSICCollate`.  Unlike
  SubspaceNet/ReconUNet the model does *not* consume the lag stack.
* Output: ``angles_pred`` ``[B, M]`` in **broadside radians** (the upstream
  network regresses radians directly; its own training loss is RMSPE on
  radians), wrapped in :class:`BaselineOutput`.

Fixed source count
------------------
The published head is ``nn.Linear(hidden, M)`` — one model per source count.
We follow the published protocol: train one instance per ``K ∈ k_choices`` on
the K-subset of the shared corpus (``data.k_filter`` in the train config) with
the *identical* optimiser schedule used for the other models, and dispatch by
the true K at evaluation.  (K is assumed known for every method in the paper.)

Deviations from the vendored forward (all documented, none change parameters)
--------------------------------------------------------------------------------
1. **Batched** MUSIC spectrum.  Upstream loops per sample and per grid angle
   (``pre_MUSIC`` / ``spectrum_calculation``: 2048 × 361 tiny matmuls per
   batch) which is unusable at 2M samples; here it is one ``einsum``.
2. **Hermitian PSD surrogate + robust eigh** (``hermitian_psd=True``,
   default).  Upstream calls ``torch.linalg.eig`` on the *general* learned
   matrix and takes eigenvector columns ``M:`` in LAPACK's unspecified order;
   ``eig``'s backward is also numerically fragile.  We form
   ``R_z = K_x^H K_x + εI`` (SubspaceNet's ``gram_diagonal_overload``) and use
   the project's phase-safe differentiable noise projector — the same
   machinery the SubspaceNet baseline runs on, so both learned baselines share
   one eigendecomposition path.  ``hermitian_psd=False`` gives a batched
   version of the upstream ``eig`` route (noise subspace = eigenvectors with
   the smallest |λ|) for reference.
3. **Time-major GRU input.**  Upstream reshapes ``[B, 2N, T] → [B, T, 2N]`` with
   ``.view`` (a memory reinterpretation, not a transpose), which scrambles the
   time axis.  We use the intended ``transpose`` so the GRU sees one 2N-vector
   per snapshot, as the DA-MUSIC paper describes.
4. Grid steering vectors / angle grid are registered as **buffers** so they
   follow ``model.to(device)`` (upstream keeps them as plain CPU tensors).

Everything else — ``BatchNorm1d(T)``, GRU sizes, ``fc → fc1 → fc2 → fc2 → fc3``
(upstream applies ``fc2`` twice; kept verbatim), ReLUs, Xavier init — is the
vendored module's own parameters and ordering.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ._base import BaselineAdapter, BaselineOutput
from .subspacenet_adapter import _ensure_on_path


class DAMUSICAdapter(BaselineAdapter):
    """Canonical-interface wrapper around ``subspacenet.src.models.DeepAugmentedMUSIC``."""

    name = "damusic"

    def __init__(
        self,
        N: int = 8,
        T: int = 512,
        M: int = 1,
        hermitian_psd: bool = True,
        psd_eps: float = 1e-6,
    ) -> None:
        """``N`` sensors, ``T`` snapshots (bound by the upstream ``BatchNorm1d(T)``),
        ``M`` sources (fixed head width)."""
        self.N = int(N)
        self.T = int(T)
        self.M = int(M)
        self.hermitian_psd = bool(hermitian_psd)
        self.psd_eps = float(psd_eps)

    # --- construction -------------------------------------------------------

    def build_model(self, cfg: Dict[str, Any]) -> nn.Module:
        _ensure_on_path()
        from src.models import DeepAugmentedMUSIC  # type: ignore  # vendored

        self.N = int(cfg.get("N", self.N))
        self.T = int(cfg.get("T", self.T))
        self.M = int(cfg.get("M", self.M))
        self.hermitian_psd = bool(cfg.get("hermitian_psd", self.hermitian_psd))
        self.psd_eps = float(cfg.get("psd_eps", self.psd_eps))

        model = DeepAugmentedMUSIC(N=self.N, T=self.T, M=self.M)

        # Deviation 4: promote the grid steering matrix [G, N] and the angle
        # grid [G] from plain attributes to buffers so they move with the model.
        sv = model.sv.detach().clone()
        angels = model.angels.detach().clone()
        del model.sv
        del model.angels
        model.register_buffer("sv", sv)
        model.register_buffer("angels", angels)
        return model

    # --- data-plane ---------------------------------------------------------

    def prepare_input(self, snapshots: torch.Tensor, meta: Dict[str, Any]) -> torch.Tensor:
        """DA-MUSIC consumes the raw complex snapshots ``[B, N, T]`` unchanged."""
        return snapshots

    # --- core ----------------------------------------------------------------

    def surrogate_covariance(self, model: nn.Module, X: torch.Tensor) -> torch.Tensor:
        """GRU + linear map → learned ``K_x`` ``[B, N, N]`` (complex), upstream steps 1–4."""
        B, N, _ = X.shape
        Xr = torch.cat((X.real, X.imag), dim=1)          # [B, 2N, T]
        Xr = Xr.transpose(1, 2).contiguous()             # [B, T, 2N]  (deviation 3)
        Xr = model.BatchNorm(Xr)                         # BatchNorm1d(T) over the T axis
        gru_out, _ = model.rnn(Xr)                       # [B, T, 2N]
        h = gru_out[:, -1]                               # [B, 2N]
        Rx = model.fc(h).view(B, 2 * N, N)               # [B, 2N, N]
        return torch.complex(Rx[:, :N, :], Rx[:, N:, :]) # [B, N, N]

    def noise_projector(self, Kx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``F = U_n U_n^H`` ``[B, N, N]`` and the matrix it was computed from."""
        N = Kx.shape[-1]
        if self.hermitian_psd:
            eye = torch.eye(N, device=Kx.device, dtype=Kx.dtype)
            Rz = Kx.conj().transpose(1, 2) @ Kx + self.psd_eps * eye   # Hermitian PSD
            from reconunet.models.deep_learning.subspace_models import _noise_projector_batched
            F = _noise_projector_batched(Rz, self.M)
            return F, Rz
        # Batched upstream-faithful route: general eig, noise = smallest |λ|.
        lam, V = torch.linalg.eig(Kx)
        order = torch.argsort(lam.abs(), dim=-1, descending=True)
        V = torch.gather(V, 2, order.unsqueeze(1).expand(-1, N, -1))
        Un = V[:, :, self.M:]
        return Un @ Un.conj().transpose(1, 2), Kx

    @staticmethod
    def music_spectrum(model: nn.Module, F: torch.Tensor) -> torch.Tensor:
        """Batched ``1 / Re(a_g^H F a_g)`` over the model's grid → ``[B, G]``.

        Numerically identical to upstream ``spectrum_calculation`` per sample
        (see ``tests/unit/test_damusic_adapter.py``).
        """
        sv = model.sv.to(F.dtype)                                      # [G, N]
        denom = torch.einsum("gn,bnm,gm->bg", sv.conj(), F, sv).real   # [B, G]
        return 1.0 / denom.clamp_min(1e-9)

    def forward(
        self,
        model: nn.Module,
        prepped: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> BaselineOutput:
        X = prepped
        if X.shape[-1] != self.T:
            raise ValueError(
                f"DA-MUSIC was built for T={self.T} snapshots (BatchNorm1d(T)); got T={X.shape[-1]}"
            )
        Kx = self.surrogate_covariance(model, X)
        F, Rz = self.noise_projector(Kx)
        spectrum = self.music_spectrum(model, F)          # [B, G]
        y = model.ReLU(model.fc1(spectrum))
        y = model.ReLU(model.fc2(y))
        y = model.ReLU(model.fc2(y))                      # upstream applies fc2 twice
        doa = model.fc3(y)                                # [B, M] broadside radians

        loss = None
        if targets is not None and self.training_loss_fn is not None:
            loss = self.training_loss_fn(doa, targets)
        # NOTE: the key is deliberately *not* "spatial_spectrum" — the trainer
        # routes that name to a grid-BCE loss (SubViT); DA-MUSIC trains on RMSPE.
        return BaselineOutput(
            angles_pred=doa, loss=loss,
            extras={"Rz": Rz, "music_spectrum": spectrum},
        )

    # --- training loss (pluggable; None ⇒ trainer applies RMSPE) -------------

    training_loss_fn = None


class DAMUSICEnsemble:
    """Per-source-count DA-MUSIC models dispatched by the true K at evaluation.

    The published DA-MUSIC head is fixed-width, so the paper-faithful protocol
    trains one instance per K (``configs/train/damusic_paper_k*.yaml``), each
    writing ``<root>/k<K>/checkpoints/best.pt``.  :meth:`from_run_dir` loads
    every K it finds; :meth:`predict` routes each sample to the model matching
    its true source count and NaN-pads to the batch's ``K_max`` — the same
    ``[B, K_max]`` NaN-padded convention the other adapters use.

    The architecture is inferred from each checkpoint's state dict (``fc3`` →
    M, ``BatchNorm`` → T, ``sv`` buffer → N), cross-checked against the saved
    training config, so evaluation never hard-codes a model shape.
    """

    def __init__(self, models: Dict[int, tuple[DAMUSICAdapter, nn.Module]]):
        if not models:
            raise ValueError("DAMUSICEnsemble needs at least one per-K model")
        self.models = dict(models)

    @property
    def ks(self) -> list[int]:
        return sorted(self.models)

    @staticmethod
    def load_single(path, device: torch.device | str = "cpu") -> tuple[DAMUSICAdapter, nn.Module]:
        """Build + load one DA-MUSIC instance from a trainer checkpoint."""
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        state = {k.removeprefix("_inner."): v for k, v in state.items()}
        init: Dict[str, Any] = {}
        if isinstance(ck, dict):
            init.update((ck.get("cfg") or {}).get("model", {}).get("init", {}) or {})
        # Shapes from the weights win over the config (they are what actually trained).
        init["M"] = int(state["fc3.weight"].shape[0])
        init["T"] = int(state["BatchNorm.weight"].shape[0])
        init["N"] = int(state["sv"].shape[1])
        keep = {k: init[k] for k in ("N", "T", "M", "hermitian_psd", "psd_eps") if k in init}
        adapter = DAMUSICAdapter(**keep)
        model = adapter.build_model(keep)
        model.load_state_dict(state, strict=True)
        return adapter, model.to(device).eval()

    @classmethod
    def from_run_dir(
        cls,
        root,
        device: torch.device | str = "cpu",
        ks: tuple[int, ...] = (1, 2, 3, 4),
        verbose: bool = True,
    ) -> Optional["DAMUSICEnsemble"]:
        """Load ``<root>/k<K>/checkpoints/best.pt`` for every available K.

        Returns ``None`` (with a printed warning) when no checkpoint exists, so
        analysis scripts can simply skip the method before the models are trained.
        """
        from pathlib import Path

        models: Dict[int, tuple[DAMUSICAdapter, nn.Module]] = {}
        for k in ks:
            p = Path(root) / f"k{k}" / "checkpoints" / "best.pt"
            if not p.is_file():
                continue
            adapter, model = cls.load_single(p, device)
            if adapter.M != k:
                raise ValueError(f"{p}: checkpoint head width M={adapter.M} but directory says K={k}")
            models[k] = (adapter, model)
        if not models:
            if verbose:
                print(f"[damusic] no per-K checkpoints under {root} (expected k<K>/checkpoints/best.pt) "
                      f"— DA-MUSIC will be skipped")
            return None
        if verbose:
            missing = [k for k in ks if k not in models]
            print(f"[damusic] loaded per-K models for K={sorted(models)}"
                  + (f" (missing K={missing} → NaN for those samples)" if missing else ""))
        return cls(models)

    @torch.no_grad()
    def predict(self, snaps: torch.Tensor, n_sources) -> torch.Tensor:
        """``snaps`` ``[g, N, T]`` complex (on the models' device), ``n_sources``
        ``[g]`` ints → angles ``[g, K_max]`` broadside radians, NaN-padded."""
        import numpy as np

        ns = (n_sources.detach().cpu().numpy() if isinstance(n_sources, torch.Tensor)
              else np.asarray(n_sources)).astype(int)
        g = snaps.shape[0]
        K_max = max(int(ns.max()), 1) if g else 1
        out = torch.full((g, K_max), float("nan"), device=snaps.device, dtype=torch.float32)
        for k in np.unique(ns):
            k = int(k)
            if k not in self.models:
                continue
            mask = torch.from_numpy(ns == k).to(snaps.device)
            adapter, model = self.models[k]
            pred = adapter.forward(model, snaps[mask]).angles_pred.to(out.dtype)
            out[mask.nonzero(as_tuple=True)[0], :k] = pred[:, :k]
        return out
