
"""Subspace-based deep-learning models for DoA estimation (ported from SubspaceNet).

This module brings the model-based deep-learning architectures that were originally
implemented in the SubspaceNet project into the Tri4Net code-base.  The original
`ModelGenerator` abstraction is **not** included - the classes can be instantiated
directly.

Implemented models
------------------
* DeepRootMUSIC          - CNN encoder that outputs a surrogate covariance fed into
                           differentiable Root-MUSIC.
* SubspaceNet            - Generalisation of DeepRootMUSIC with an anti-rectifier and
                           selectable differentiable subspace method (Root-MUSIC / ESPRIT).
* SubspaceNetEsprit      - Convenience subclass of SubspaceNet that always uses ESPRIT.
* DeepAugmentedMUSIC     - GRU based network that produces a MUSIC spectrum which is
                           post-processed by a small MLP.
* DeepCNN                - Baseline CNN that maps covariance-like inputs directly to
                           a probability grid over angles.

All models inherit from ``torch.nn.Module``.  If Tri4Net's ``ModelRegistry`` is
available, the classes are also automatically registered so they can be created
from configuration files exactly like the existing deep-learning models.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Optional Tri4Net integration -------------------------------------------------
# -----------------------------------------------------------------------------
try:
    from ..model_registry import ModelRegistry  # type: ignore
except Exception:  # pragma: no cover – registry is optional
    ModelRegistry = None  # noqa: N816 – keep camel case to match Tri4Net style

# -----------------------------------------------------------------------------
# Utility helpers --------------------------------------------------------------
# -----------------------------------------------------------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def sum_of_diags_torch(matrix: torch.Tensor) -> torch.Tensor:
    """Sum the diagonals of a square matrix (PyTorch version).

    Returns the sums ordered from the *lower* diagonals to the *upper* diagonals,
    i.e. equivalent to ``numpy`` with ``offset`` ranging from ``-(N-1)`` to
    ``+(N-1)``.
    """
    diag_sum = []
    diag_index = torch.linspace(
        -matrix.shape[-1] + 1, matrix.shape[-1] - 1, 2 * matrix.shape[-1] - 1, dtype=torch.int64
    )
    for idx in diag_index:
        diag_sum.append(torch.sum(torch.diagonal(matrix, offset=int(idx))) )
    return torch.stack(diag_sum, dim=0)


def sum_of_diags_batched(matrix: torch.Tensor) -> torch.Tensor:
    """Batched diagonal sums for ``[B, N, N]`` → ``[B, 2N-1]``."""
    B, N, _ = matrix.shape
    n_diags = 2 * N - 1
    result = []
    for offset in range(-(N - 1), N):
        result.append(torch.diagonal(matrix, offset=offset, dim1=-2, dim2=-1).sum(-1))
    return torch.stack(result, dim=-1)  # [B, 2N-1]


def find_roots_torch(coefficients: torch.Tensor) -> torch.Tensor:
    """Solve for the roots of a polynomial with the given *coefficients*.

    The companion-matrix approach is used to support back-propagation.
    """
    # Companion matrix
    n = len(coefficients) - 1
    A = torch.zeros((n, n), dtype=coefficients.dtype, device=coefficients.device)
    A[1:, :-1] = torch.eye(n - 1, dtype=coefficients.dtype, device=coefficients.device)
    A[0, :] = -coefficients[1:] / coefficients[0]
    return torch.linalg.eigvals(A)


def find_roots_batched(coefficients: torch.Tensor) -> torch.Tensor:
    """Batched polynomial root-finding via companion matrix eigvals.

    ``coefficients`` has shape ``[B, D+1]``; returns ``[B, D]`` complex roots.
    """
    B, D1 = coefficients.shape
    n = D1 - 1
    eye = torch.eye(n - 1, dtype=coefficients.dtype, device=coefficients.device)
    eye = eye.unsqueeze(0).expand(B, -1, -1)                   # [B, n-1, n-1]
    A = torch.zeros(B, n, n, dtype=coefficients.dtype, device=coefficients.device)
    A[:, 1:, :-1] = eye
    A[:, 0, :] = -coefficients[:, 1:] / coefficients[:, :1]    # broadcast c0
    return torch.linalg.eigvals(A)                              # [B, n]


def gram_diagonal_overload(
    Kx: torch.Tensor, eps: float = 1e-6, batch_size: Optional[int] = None
) -> torch.Tensor:
    """Create a Hermitian, positive semi-definite surrogate covariance matrix.

    The output is :math:`R_z = K_x^H K_x + \epsilon I` computed batch-wise.
    """
    if not isinstance(Kx, torch.Tensor):
        Kx = torch.as_tensor(Kx)

    if batch_size is None:
        batch_size = Kx.shape[0]

    eye = None  # lazily allocated identity (on correct device / dtype)
    Rz_list = []
    for idx in range(batch_size):
        K = Kx[idx]
        K_gram = (K.conj().T @ K).to(device)
        if eye is None:
            eye = torch.eye(K_gram.shape[0], device=K_gram.device, dtype=K_gram.dtype)
        Rz_list.append(K_gram + eps * eye)
    return torch.stack(Rz_list, dim=0)

# -----------------------------------------------------------------------------
# Differentiable sub-space methods --------------------------------------------
# -----------------------------------------------------------------------------

# Numerical-stability constants for the differentiable eigendecomposition.
_EIGH_REL_LOAD = 1e-6        # relative diagonal load (× mean |diag(R)|)
_EIGH_ABS_LOAD = 1e-9        # absolute floor so an all-zero R is still PD
_EIGH_MAX_RETRIES = 4        # escalate the load ×100 each retry on failure
_EIGH_BACKWARD_DAMP = 1e-6   # δ in the damped reciprocal  diff / (diff² + δ²)

# Count of eigh inputs that had to be sanitised (non-finite entries replaced).
# Surfaced via a rate-limited warning so an isolated bad sample in 2M is quiet
# but a genuine training divergence (every batch non-finite) is loud.
_EIGH_NONFINITE_COUNT = 0


def _warn_nonfinite_eigh(n_bad: int) -> None:
    """Rate-limited warning when an eigh input contains NaN/Inf.

    Warns on the 1st, 10th, 100th, … occurrence so a rare degenerate covariance
    stays quiet while a runaway divergence (sanitisation firing every batch)
    becomes impossible to miss in the log.
    """
    global _EIGH_NONFINITE_COUNT
    _EIGH_NONFINITE_COUNT += 1
    c = _EIGH_NONFINITE_COUNT
    if c == 1 or c % 10 == 0 and c < 100 or c % 100 == 0:
        warnings.warn(
            f"_robust_hermitian_eigh: sanitised a non-finite covariance "
            f"({n_bad} bad entries); occurrence #{c}. Isolated events are "
            f"harmless (degenerate sample); a steadily rising count signals "
            f"training divergence.",
            RuntimeWarning, stacklevel=2,
        )


def _eigh_qr_fallback(loaded: torch.Tensor):
    """Per-matrix eigendecomposition for Hermitian inputs that defeat LAPACK's
    batched divide-and-conquer solver (``heevd`` — "failed to converge … too many
    repeated eigenvalues", error code 7).

    A *uniform* diagonal load cannot rescue that failure mode: it shifts every
    eigenvalue equally, so the eigenvalue **gaps** (what the divide-and-conquer
    routine struggles with on clustered spectra) are untouched.  This fallback
    instead retries each matrix individually with torch's eigh and, for any that
    still fail, the globally-convergent QR driver (``heev``/``syev`` via SciPy,
    ``driver='ev'``), which always converges for a Hermitian matrix.  Slow but
    only reached on the rare offending batch, so the throughput cost is nil.

    ``loaded`` is the already-diagonal-loaded CPU ``complex128`` tensor, shape
    ``[..., N, N]``.  Returns ``(eigenvalues_ascending, U)`` matching
    :func:`torch.linalg.eigh`'s convention and dtypes.
    """
    import scipy.linalg as _sla                      # lazy: only on the rare path

    N = loaded.shape[-1]
    flat = loaded.reshape(-1, N, N)
    # Belt-and-suspenders: the primary sanitisation lives in
    # _robust_hermitian_eigh, but guard here too so a direct caller can't hit
    # scipy's "array must not contain infs or NaNs" ValueError.
    if not torch.isfinite(flat).all():
        flat = torch.complex(
            torch.nan_to_num(flat.real, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(flat.imag, nan=0.0, posinf=0.0, neginf=0.0),
        )
    w_out = torch.empty(flat.shape[0], N, dtype=torch.float64)
    v_out = torch.empty_like(flat)
    for i in range(flat.shape[0]):
        A = flat[i]
        try:
            w_i, v_i = torch.linalg.eigh(A)          # per-matrix: isolates the bad one
        except RuntimeError:                         # incl. torch._C._LinAlgError
            # QR iteration is globally convergent for Hermitian A (UPLO='L' to
            # match torch.linalg.eigh, which reads the lower triangle).
            w_np, v_np = _sla.eigh(A.numpy(), driver="ev", lower=True)
            w_i = torch.from_numpy(np.ascontiguousarray(w_np))
            v_i = torch.from_numpy(np.ascontiguousarray(v_np))
        w_out[i] = w_i.to(torch.float64)
        v_out[i] = v_i.to(torch.complex128)
    return w_out.reshape(*loaded.shape[:-1]), v_out.reshape(loaded.shape)


def _robust_hermitian_eigh(R: torch.Tensor):
    """Eigendecomposition of (a batch of) Hermitian matrices, robust to the
    degenerate / ill-conditioned inputs produced early in training.

    Computed on CPU in double precision — LAPACK is both faster and far more
    robust than cuSOLVER for the small (e.g. 8×8) matrices SubspaceNet emits.
    A per-sample **uniform** diagonal load ``c·I`` is added first; because a
    uniform shift moves every eigenvalue equally it leaves the eigenvectors —
    and therefore the downstream noise-subspace projector, polynomial roots and
    DoA estimates — exactly unchanged, so the load may be as large as numerical
    robustness requires *for free*.  On failure the load is escalated and the
    decomposition retried (this is what prevents the historic
    ``linalg.eigh failed to converge`` crash on near-degenerate covariances).

    ``R`` has shape ``[..., N, N]``.  Returns ``(eigenvalues, U)`` with
    eigenvalues **ascending** (LAPACK convention), shapes ``[..., N]`` / ``[..., N, N]``.
    """
    N = R.shape[-1]
    R_cpu = R.detach().cpu().to(torch.complex128)
    # Defensive sanitisation: a pathological sample (fully-coherent multipath,
    # or a transient training instability) can yield a covariance with NaN/Inf
    # entries.  LAPACK/scipy raise an *uncaught* ValueError on non-finite input
    # ("array must not contain infs or NaNs"), which historically killed
    # multi-day runs over a single bad matrix in 2M.  Replace non-finite entries
    # with 0 (real & imag separately — torch.nan_to_num rejects complex) so the
    # diagonal load below makes the matrix well-posed and the eigh degrades
    # gracefully.  No-op on the overwhelmingly common all-finite path.
    if not torch.isfinite(R_cpu).all():
        _warn_nonfinite_eigh(int((~torch.isfinite(R_cpu)).sum()))
        R_cpu = torch.complex(
            torch.nan_to_num(R_cpu.real, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(R_cpu.imag, nan=0.0, posinf=0.0, neginf=0.0),
        )
    eye = torch.eye(N, dtype=torch.complex128)
    # Per-sample scale = mean magnitude of the (real) diagonal of R.
    diag = R_cpu.diagonal(dim1=-2, dim2=-1).real             # [..., N]
    scale = diag.abs().mean(dim=-1)                          # [...]
    load = _EIGH_REL_LOAD * scale + _EIGH_ABS_LOAD           # [...]
    for attempt in range(_EIGH_MAX_RETRIES + 1):
        loaded = R_cpu + load[..., None, None] * eye
        try:
            return torch.linalg.eigh(loaded)
        except RuntimeError:                                 # incl. torch._C._LinAlgError
            if attempt == _EIGH_MAX_RETRIES:
                # Escalating the uniform load can't fix a divide-and-conquer
                # convergence failure on clustered/repeated eigenvalues, so a
                # plain re-raise here is what historically killed multi-hour
                # runs (one degenerate covariance in 2M samples).  Fall back
                # per-matrix to a globally-convergent QR solver instead.
                return _eigh_qr_fallback(loaded)
            load = load * 100.0


class _DiffNoiseProjectorBatched(torch.autograd.Function):
    """Batched differentiable noise-subspace projector ``F = Un @ Un^H``.

    Forward: ``torch.linalg.eigh`` on CPU (LAPACK, robust) → select noise
    eigenvectors → projector.  CPU is both faster and more robust than
    cuSOLVER for small (8×8) Hermitian matrices.
    Backward: analytical spectral-perturbation gradient (fully batched, GPU).

    Input ``R`` has shape ``[B, N, N]`` (batch of Hermitian matrices).
    Returns ``F`` of the same shape.
    """

    @staticmethod
    def forward(ctx, R: torch.Tensor, n_signal: int) -> torch.Tensor:
        orig_device = R.device
        # Robust CPU/double eigh with per-sample relative diagonal loading and
        # retry-on-failure (see :func:`_robust_hermitian_eigh`).  Returns
        # ascending eigenvalues; flip to descending so columns [:M] are signal.
        eigenvalues_cpu, U_cpu = _robust_hermitian_eigh(R)
        # Keep eigenvalues in the real dtype matching R (float32 for the
        # production complex64 path; float64 when called in double precision,
        # e.g. autograd gradcheck).  Flip ascending→descending.
        _real_dtype = torch.float64 if R.dtype == torch.complex128 else torch.float32
        eigenvalues = eigenvalues_cpu.to(device=orig_device, dtype=_real_dtype).flip(-1)
        U = U_cpu.to(dtype=R.dtype, device=orig_device).flip(-1)
        ctx.save_for_backward(eigenvalues, U)
        ctx.n_signal = n_signal
        Un = U[:, :, n_signal:]                         # [B, N, N-M]
        return Un @ Un.conj().mT                        # [B, N, N]

    @staticmethod
    def backward(ctx, grad_F: torch.Tensor):
        eigenvalues, U = ctx.saved_tensors              # [B,N], [B,N,N]
        M = ctx.n_signal

        # F = Un Un^H is structurally Hermitian, so only the Hermitian part of
        # the incoming cotangent affects the true gradient (dF is Hermitian).
        # Symmetrise so the VJP is correct for any upstream cotangent.
        grad_F = 0.5 * (grad_F + grad_F.conj().mT)

        G = U.conj().mT @ grad_F @ U                   # [B, N, N]  (Hermitian)

        lam = eigenvalues                               # [B, N]
        diffs = lam.unsqueeze(-1) - lam.unsqueeze(-2)   # [B, N, N]; diffs[i,j]=λ_i-λ_j
        # Damped reciprocal in place of a hard floor: bounded by 1/(2δ) and
        # smoothly → 0 as eigenvalues collapse, so near-degenerate spectra no
        # longer blow up the gradient.
        recip = diffs / (diffs * diffs + _EIGH_BACKWARD_DAMP ** 2)

        # VJP of the noise-subspace projector F = Un Un^H.  Only the
        # signal↔noise cross-terms survive (intra-subspace rotations cancel):
        #   C[i,j] = G[i,j] / (λ_i - λ_j)   for i∈noise, j∈signal.
        # The gradient w.r.t. a Hermitian R must itself be Hermitian, so the
        # signal-row/noise-col block is the conjugate-transpose of the
        # noise-row/signal-col block — completing it that way is what makes the
        # analytic gradient match finite differences (see gradcheck test).
        coeff = torch.zeros_like(G)
        coeff[:, M:, :M] = G[:, M:, :M] * recip[:, M:, :M]
        coeff[:, :M, M:] = coeff[:, M:, :M].conj().transpose(-2, -1)

        return U @ coeff @ U.conj().mT, None


def _noise_projector_batched(R: torch.Tensor, n_signal: int) -> torch.Tensor:
    """Phase-safe differentiable noise projector (batched)."""
    return _DiffNoiseProjectorBatched.apply(R, n_signal)


# Keep single-sample wrapper for backward compat / esprit
def _noise_projector(R: torch.Tensor, n_signal: int) -> torch.Tensor:
    """Phase-safe differentiable noise projector ``F = Un @ Un^H``."""
    return _noise_projector_batched(R.unsqueeze(0), n_signal).squeeze(0)


class _DiffEspritPhi(torch.autograd.Function):
    """Differentiable ESPRIT shift-invariance matrix ``Φ = pinv(Us↑) @ Us↓``.

    Like :class:`_DiffNoiseProjector`, Φ is phase-invariant (the column
    phases of Us cancel between pinv and the product), so its gradient
    w.r.t. R is well-defined even though individual eigenvectors are not.
    """

    @staticmethod
    def forward(ctx, R: torch.Tensor, n_signal: int) -> torch.Tensor:
        orig_device = R.device
        # Robust CPU/double eigh (single Hermitian matrix here); see
        # :func:`_robust_hermitian_eigh`.  Flip to descending.
        eigenvalues_cpu, U_cpu = _robust_hermitian_eigh(R)
        # Keep eigenvalues in the real dtype matching R (float32 for the
        # production complex64 path; float64 when called in double precision,
        # e.g. autograd gradcheck).  Flip ascending→descending.
        _real_dtype = torch.float64 if R.dtype == torch.complex128 else torch.float32
        eigenvalues = eigenvalues_cpu.to(device=orig_device, dtype=_real_dtype).flip(-1)
        U = U_cpu.to(dtype=R.dtype, device=orig_device).flip(-1)
        Us = U[:, :n_signal]
        Us_upper, Us_lower = Us[:-1], Us[1:]
        phi = torch.linalg.pinv(Us_upper) @ Us_lower
        ctx.save_for_backward(eigenvalues, U, phi)
        ctx.n_signal = n_signal
        return phi

    @staticmethod
    def backward(ctx, grad_phi: torch.Tensor):
        # NOTE: this backward is APPROXIMATE — it omits the intra-signal
        # eigenvector-coupling block (k, j both in the signal subspace) and uses
        # a simplified pseudo-inverse derivative.  It is good enough to train the
        # secondary ESPRIT head but is not finite-difference-exact (see the
        # ``xfail`` gradcheck in tests/unit/test_subspacenet_correctness.py).
        # root_music (the paper default) has an exact, validated backward.
        eigenvalues, U, phi = ctx.saved_tensors
        M = ctx.n_signal
        N = eigenvalues.shape[-1]

        Us = U[:, :M]
        Us_upper = Us[:-1]
        pinv_upper = torch.linalg.pinv(Us_upper)

        # dL/dUs_lower = pinv_upper^H @ grad_phi
        grad_Us_lower = pinv_upper.conj().T @ grad_phi
        # dL/dUs_upper = -pinv_upper^H @ grad_phi @ Us_lower^H @ pinv_upper^H
        #              = -grad_Us_lower @ (Us_lower^H @ pinv_upper^H)
        Us_lower = Us[1:]
        grad_Us_upper = -grad_Us_lower @ Us_lower.conj().T @ pinv_upper.conj().T

        # Assemble dL/dUs by adding the upper/lower contributions.
        grad_Us = torch.zeros_like(Us)
        grad_Us[:-1] += grad_Us_upper
        grad_Us[1:] += grad_Us_lower

        # Now propagate dL/dUs → dL/dR via spectral perturbation.
        # G_full = U^H (dL/dU_full) where dL/dU_full has signal columns only.
        grad_U_full = torch.zeros_like(U)
        grad_U_full[:, :M] = grad_Us
        G = U.conj().T @ grad_U_full

        lam = eigenvalues
        diffs = lam.unsqueeze(-1) - lam.unsqueeze(-2)
        # Damped reciprocal (bounded, → 0 on degeneracy); see _DiffNoiseProjectorBatched.
        recip = diffs / (diffs * diffs + _EIGH_BACKWARD_DAMP ** 2)

        coeff = torch.zeros_like(G)
        # Signal-noise cross terms for dU_signal/dR.
        coeff[:M, M:] = G[:M, M:] * recip[:M, M:]
        coeff[M:, :M] = G[M:, :M] * recip[M:, :M]

        return U @ coeff @ U.conj().T, None


def _esprit_phi(R: torch.Tensor, n_signal: int) -> torch.Tensor:
    """Phase-safe differentiable ESPRIT Φ matrix."""
    return _DiffEspritPhi.apply(R, n_signal)


def root_music(Rz: torch.Tensor, M: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable Root-MUSIC implementation (narrow-band ULA, ideal case).

    Fully batched: one eigh call, one eigvals call, and a **vectorised**
    root-selection (no per-sample Python loop) — the M roots inside the unit
    circle and closest to it are picked with a batched argsort/gather, which is
    what makes large-corpus (2M-sample) training tractable.
    """
    element_spacing = 0.5  # wavelength units
    N = Rz.shape[-1]
    k_d = element_spacing * 2 * np.pi
    c = N / 2 if N % 2 == 0 else (N - 1) / 2
    _rad2deg = 180.0 / np.pi

    # Batched noise projector — single eigh call for whole batch
    F_batch = _noise_projector_batched(Rz, M)               # [B, N, N]

    # Batched diagonal sums → polynomial coefficients
    coeffs_batch = sum_of_diags_batched(F_batch)             # [B, 2N-1]

    # Batched polynomial root-finding — single eigvals call
    roots_batch = find_roots_batched(coeffs_batch)           # [B, 2N-2]

    # Shift roots for array center
    phase = torch.tensor(1j * k_d * c, dtype=roots_batch.dtype, device=roots_batch.device)
    roots_shifted = roots_batch * torch.exp(phase)           # [B, 2N-2]

    # All roots → DoA (for debugging / spectrum)
    roots_angles_all = torch.angle(roots_shifted) / k_d
    roots_angles_all = torch.clamp(roots_angles_all, min=-1.0, max=1.0)
    doa_all = torch.acos(-roots_angles_all) * _rad2deg       # [B, 2N-2]

    # Vectorised root selection (replaces the per-sample Python loop): for each
    # sample pick the M roots INSIDE the unit circle and closest to it.  Score
    # inside roots by their distance to the circle and push outside roots to +inf
    # so the M smallest scores are exactly "inside & closest" — identical to the
    # old "sort-by-distance → keep-inside → take-M" logic, but fully batched.
    mag = torch.abs(roots_shifted)                            # [B, 2N-2]
    dist = torch.abs(mag - 1.0)                               # distance to unit circle
    inf = torch.full_like(dist, float("inf"))
    score = torch.where(mag < 1.0, dist, inf)                 # [B, 2N-2]
    sel_idx = torch.argsort(score, dim=-1)[:, :M]             # [B, M]
    # Gather the per-root DoA (real, differentiable; same angle→acos map as doa_all).
    doa_pred = torch.gather(doa_all, 1, sel_idx)              # [B, M]

    return (
        doa_pred,
        doa_all,
        roots_batch[-1],  # last sample's roots for debugging
    )


def esprit(Rz: torch.Tensor, M: int, batch_size: int) -> torch.Tensor:
    """Differentiable ESPRIT implementation (narrow-band ULA, ideal case)."""
    doa_batches = []
    for b in range(batch_size):
        R = Rz[b]
        # Differentiable ESPRIT Phi via custom autograd Function.
        phi = _esprit_phi(R, M)
        phi_eigs = torch.linalg.eigvals(phi)  # Only eigenvalues, avoids eigenvector gradient issues
        angles = torch.angle(phi_eigs)
        doa_batches.append(-torch.arcsin(angles / np.pi))
    return torch.stack(doa_batches, dim=0)

# -----------------------------------------------------------------------------
# Model definitions ------------------------------------------------------------
# -----------------------------------------------------------------------------


class _AntiRectifier(nn.Module):
    """Helper layer that concatenates ReLU(x) and ReLU(-x) along channel dim."""

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401  – simple forward
        return torch.cat((self.relu(x), self.relu(-x)), dim=1)


# =============================================================================
# DeepRootMUSIC ----------------------------------------------------------------
# =============================================================================


class DeepRootMUSIC(nn.Module):
    """CNN encoder followed by differentiable Root-MUSIC."""

    def __init__(self, tau: int, M: int, activation_value: float = 0.1):
        super().__init__()
        self.tau = tau
        self.M = M

        self.conv1 = nn.Conv2d(self.tau, 16, kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2)

        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2)
        self.deconv3 = nn.ConvTranspose2d(16, 1, kernel_size=2)

        self.act = nn.LeakyReLU(activation_value)
        self.drop = nn.Dropout(0.2)

    def forward(self, Rx_tau: torch.Tensor):
        # Input: [B, tau, 2N, N]
        N = Rx_tau.shape[-1]
        B = Rx_tau.shape[0]

        x = self.act(self.conv1(Rx_tau))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.deconv1(x))
        x = self.act(self.deconv2(x))
        x = self.drop(x)
        Rx = self.deconv3(x)  # [B, 1, 2N, N]

        Rx_view = Rx.view(B, Rx.size(2), Rx.size(3))  # [B, 2N, N]
        Rx_real, Rx_imag = Rx_view[:, :N, :], Rx_view[:, N:, :]
        Kx_tag = torch.complex(Rx_real, Rx_imag)

        Rz = gram_diagonal_overload(Kx_tag)
        doa_pred, doa_all, roots = root_music(Rz, self.M, B)
        return doa_pred, doa_all, roots, Rz


# =============================================================================
# SubspaceNet ------------------------------------------------------------------
# =============================================================================


class SubspaceNet(nn.Module):
    """Generalised subspace-method network supporting Root-MUSIC or ESPRIT.

    .. deprecated::
        The canonical SubspaceNet used for the paper comparison is the **upstream
        submodule** model (``third_party/subspacenet/src/models.py::SubspaceNet``),
        instantiated and wrapped by
        :class:`reconunet.models.third_party.subspacenet_adapter.SubspaceNetAdapter`
        (which monkey-patches in the differentiable :func:`root_music` / :func:`esprit`
        heads from this module).  This local re-implementation is retained only for
        the :class:`ModelRegistry` / legacy training-example paths and must not drift
        from upstream — prefer the adapter.  Do **not** add new behaviour here; the
        reusable pieces (``root_music``, ``esprit``, the autograd ``Function``\\ s and
        ``gram_diagonal_overload``) live at module scope and are shared with the adapter.
    """

    def __init__(self, tau: int, M: int, diff_method: str = "root_music"):
        super().__init__()
        self.tau = tau
        self.M = M

        self.conv1 = nn.Conv2d(self.tau, 16, kernel_size=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2)

        self.deconv1 = nn.ConvTranspose2d(128, 32, kernel_size=2)
        self.deconv2 = nn.ConvTranspose2d(64, 16, kernel_size=2)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=2)

        self.drop = nn.Dropout(0.2)
        self.anti_rect = _AntiRectifier()

        self.set_diff_method(diff_method)

    # ---------------------------------------------------------------------
    # Configuration helpers ------------------------------------------------
    # ---------------------------------------------------------------------
    def set_diff_method(self, diff_method: str = "root_music"):
        if diff_method.lower().startswith("root_music"):
            self._diff_method = root_music
        elif diff_method.lower().startswith("esprit"):
            self._diff_method = esprit
        else:
            raise ValueError(f"Unsupported diff_method '{diff_method}'. Use 'root_music' or 'esprit'.")

    # ---------------------------------------------------------------------
    # Forward --------------------------------------------------------------
    # ---------------------------------------------------------------------
    def forward(self, Rx_tau: torch.Tensor):
        # Input: [B, tau, 2N, N]
        N = Rx_tau.shape[-1]
        B = Rx_tau.shape[0]

        x = self.anti_rect(self.conv1(Rx_tau))
        x = self.anti_rect(self.conv2(x))
        x = self.anti_rect(self.conv3(x))
        x = self.anti_rect(self.deconv1(x))
        x = self.anti_rect(self.deconv2(x))
        x = self.drop(x)
        Rx = self.deconv3(x)  # [B, 1, 2N, N]

        Rx_view = Rx.view(B, Rx.size(2), Rx.size(3))  # [B, 2N, N]
        Rx_real, Rx_imag = Rx_view[:, :N, :], Rx_view[:, N:, :]
        Kx_tag = torch.complex(Rx_real, Rx_imag)

        Rz = gram_diagonal_overload(Kx_tag, batch_size=B)

        output = self._diff_method(Rz, self.M, B)
        if isinstance(output, tuple):
            doa_pred, doa_all, roots = output
        else:  # ESPRIT returns only doa_pred
            doa_pred, doa_all, roots = output, None, None
        return doa_pred, doa_all, roots, Rz


# =============================================================================
# Convenience subclass for ESPRIT --------------------------------------------
# =============================================================================


class SubspaceNetEsprit(SubspaceNet):
    """SubspaceNet variant hard-wired to ESPRIT."""

    def __init__(self, tau: int, M: int):
        super().__init__(tau=tau, M=M, diff_method="esprit")

    def forward(self, Rx_tau: torch.Tensor):  # type: ignore[override]
        doa_pred, _, _, Rz = super().forward(Rx_tau)
        return doa_pred, Rz


# =============================================================================
# DeepAugmentedMUSIC -----------------------------------------------------------
# =============================================================================


class DeepAugmentedMUSIC(nn.Module):
    """GRU-based surrogate covariance → MUSIC spectrum → MLP peak detector."""

    def __init__(self, N: int, T: int, M: int):
        super().__init__()
        self.N, self.T, self.M = N, T, M
        self.angles = torch.linspace(-np.pi / 2, np.pi / 2, 361)

        self.input_size = 2 * N
        self.hidden_size = 2 * N

        self.rnn = nn.GRU(self.input_size, self.hidden_size, batch_first=True)
        self.fc_cov = nn.Linear(self.hidden_size, self.hidden_size * N)

        self.fc1 = nn.Linear(self.angles.numel(), self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, M)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.25)
        self.bn = nn.BatchNorm1d(self.T)

        self.register_buffer("_sv", self._build_steering_vectors())

        # Weight init
        for layer in [self.fc_cov, self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)

    # ------------------------------------------------------------------
    def _build_steering_vectors(self):
        sv = []
        n_idx = torch.linspace(0, self.N - 1, self.N)
        for ang in self.angles:
            sv.append(torch.exp(-1j * np.pi * n_idx * torch.sin(ang)))
        return torch.stack(sv, dim=0)  # [361, N]

    def _music_spectrum(self, Rz_batch: torch.Tensor):
        spectra = []
        for R in Rz_batch:  # loop over batch (usually small)
            _, V = torch.linalg.eigh(R)
            Un = V[:, self.M:]
            spec_vals = []
            for i in range(self.angles.numel()):
                a = self._sv[i]
                spec_vals.append(torch.real((a.conj() @ Un @ Un.conj().T @ a)))
            spec_vals = torch.stack(spec_vals)
            spectra.append(1.0 / spec_vals)
        return torch.stack(spectra, dim=0)  # [B, 361]

    # ------------------------------------------------------------------
    def forward(self, X: torch.Tensor):
        # Input X: [B, N, T] complex (torch.cfloat)
        B = X.size(0)
        X_real_imag = torch.cat((X.real, X.imag), dim=1)  # [B, 2N, T]
        X_rnn_in = X_real_imag.permute(0, 2, 1)  # [B, T, 2N]
        X_norm = self.bn(X_rnn_in)

        rnn_out, _ = self.rnn(X_norm)
        Rx_flat = self.fc_cov(rnn_out[:, -1])  # [B, 2N^2]
        Rx_view = Rx_flat.view(B, 2 * self.N, self.N)
        Rx_real = Rx_view[:, : self.N]
        Rx_imag = Rx_view[:, self.N :]
        Kx_tag = torch.complex(Rx_real, Rx_imag)

        spectrum = self._music_spectrum(Kx_tag)  # [B, 361]
        y = self.relu(self.fc1(spectrum))
        y = self.relu(self.fc2(y))
        y = self.relu(self.fc2(y))
        doa = self.fc3(y)  # [B, M]
        return doa


# =============================================================================
# DeepCNN (simple baseline) ----------------------------------------------------
# =============================================================================


class DeepCNN(nn.Module):
    """Baseline CNN for DoA probability grid prediction (copied from SubspaceNet)."""

    def __init__(self, N: int, grid_size: int):
        super().__init__()
        self.N = N
        self.grid_size = grid_size

        self.conv1 = nn.Conv2d(3, 256, kernel_size=3)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2)

        self.fc1 = nn.Linear(256 * (N - 5) * (N - 5), 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, grid_size)

        self.bn = nn.BatchNorm2d(256)
        self.drop = nn.Dropout(0.3)
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, X: torch.Tensor):
        # X: [B, N, N, 3]
        X = X.permute(0, 3, 2, 1)  # -> [B, 3, N, N]
        X = self.act(self.conv1(X))
        X = self.act(self.conv2(X))
        X = self.act(self.conv2(X))
        X = self.act(self.conv2(X))
        X = X.view(X.size(0), -1)
        X = self.drop(self.act(self.fc1(X)))
        X = self.drop(self.act(self.fc2(X)))
        X = self.drop(self.act(self.fc3(X)))
        X = self.sigmoid(self.fc4(X))
        return X


# -----------------------------------------------------------------------------
# Optional registration with Tri4Net -----------------------------------------
# -----------------------------------------------------------------------------
# UNet-based SubspaceNet variant
# -----------------------------------------------------------------------------

def anti_rect(x: torch.Tensor) -> torch.Tensor:
    """Anti-rectifier used in the paper: concat(ReLU(x), ReLU(-x))."""
    return torch.cat((F.relu(x), F.relu(-x)), 1)


class ConvBlock(nn.Module):
    """Convolution → GroupNorm → Anti-Rectifier while keeping channel count.

    We want to apply the *anti-rectifier* (concat(ReLU(x), ReLU(-x))) but keep
    the **output** channel dimension equal to ``cout`` so that the surrounding
    network architecture remains unchanged. We therefore halve the
    intermediate channel width and let the anti-rectifier double it back.
    """

    def __init__(self, cin: int, cout: int, k: int = 3, stride: int = 1, groups: int = 4):
        super().__init__()
        if cout % 2 != 0:
            raise ValueError("ConvBlock: 'cout' must be even to use anti-rectifier while keeping dimensions.")

        pad = k // 2
        inter_channels = cout // 2  # will double back to `cout` after anti-rect

        self.conv = nn.Conv2d(cin, inter_channels, k, stride=stride, padding=pad, bias=False)
        # Ensure the number of groups divides channels for GroupNorm
        gn_groups = min(groups, inter_channels)
        self.norm = nn.GroupNorm(num_groups=gn_groups, num_channels=inter_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return anti_rect(x)  # doubles channels back to `cout`


class DownBlock(nn.Module):
    """Strided conv 2x2 to down-sample."""
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = ConvBlock(cin, cout, k=3, stride=2)
    
    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    """Transpose-conv 2×2 up-sample followed by anti-rectifier (channel-safe)."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        if cout % 2 != 0:
            raise ValueError("UpBlock: 'cout' must be even when using anti-rectifier.")

        # Produce half of the desired channels, anti-rectifier will double them.
        self.up = nn.ConvTranspose2d(cin, cout // 2, kernel_size=2, stride=2, bias=False)
        gn_groups = min(4, cout // 2)
        self.norm = nn.GroupNorm(num_groups=gn_groups, num_channels=cout // 2)

    def forward(self, x):
        x = self.up(x)
        x = self.norm(x)
        return anti_rect(x)


class SubspaceUNet(nn.Module):
    """
    A light 3-level U-Net (~102k parameters for τ=8, N=8) that
    produces the surrogate covariance K̂ used by Root-MUSIC/ESPRIT.
    Fully compatible with the original SubspaceNet interface.
    Adapted for Tri4Net with 0-180 degree angle support.
    """

    def __init__(self, tau: int, M: int, diff_method: str = "root_music"):
        super().__init__()
        self.tau, self.M = tau, M

        ch = 16                     # you can halve this to 8 for a ~40-k-param variant
        # ─ encoder ─
        self.enc0  = ConvBlock(tau,      ch)
        self.down1 = DownBlock(ch,       ch*2)   # 32
        self.enc1  = ConvBlock(ch*2,     ch*2)
        self.down2 = DownBlock(ch*2,     ch*4)   # 64
        self.bott  = ConvBlock(ch*4,     ch*4)
        # ─ decoder ─
        self.up1   = UpBlock(ch*4,       ch*2)
        self.dec1  = ConvBlock(ch*4,     ch*2)   # cat → 64
        self.up2   = UpBlock(ch*2,       ch)
        self.dec2  = ConvBlock(ch*2,     ch)     # cat → 32
        self.head  = nn.Conv2d(ch, 2, kernel_size=1, bias=True)  # real & imag

        self.drop  = nn.Dropout(0.2)

        self.set_diff_method(diff_method)

    def set_diff_method(self, diff_method: str = "root_music"):
        """Set the differentiable subspace method."""
        if diff_method.lower().startswith("root_music"):
            self._diff_method = root_music
        elif diff_method.lower().startswith("esprit"):
            self._diff_method = esprit
        else:
            raise ValueError(
                f"Unsupported diff_method '{diff_method}'. "
                "Use 'root_music' or 'esprit'."
            )

    def _enforce_hermitian(self, K: torch.Tensor):
        """
        Force Hermitian symmetry: K ← (K + Kᴴ) / 2, keep diagonal real.
        K : [B, N, N] complex
        """
        K_h = K.conj().transpose(-2, -1)
        K  = 0.5 * (K + K_h)
        K.real.diagonal(dim1=-2, dim2=-1).copy_(K.real.diagonal(dim1=-2, dim2=-1))
        K.imag.diagonal(dim1=-2, dim2=-1).zero_()
        return K

    def forward(self, Rx_tau: torch.Tensor):
        """
        Input : Rx_tau  [B, τ, 2N, N]
        Output: (doa_pred, doa_all, roots, Rz)
        """
        B, _, H, W = Rx_tau.shape           # H = 2N, W = N
        N = W

        # ─── U-Net ────────────────────────────────────────────────
        x0 = self.enc0(Rx_tau)              # [B,16,H,W]
        x1 = self.enc1(self.down1(x0))      # [B,32,H/2,W/2]
        x  = self.bott(self.down2(x1))      # [B,64,H/4,W/4]

        x  = self.up1(x)                    # [B,32,H/2,W/2]
        x  = self.dec1(torch.cat([x, x1], 1))
        x  = self.up2(x)                    # [B,16,H,W]
        x  = self.dec2(torch.cat([x, x0], 1))
        x  = self.drop(x)
        head = self.head(x)                 # [B,2,H,W]

        # ─── convert to complex K̂  ───────────────────────────────
        K_r, K_i = head[:,0], head[:,1]                  # [B,H,W] each
        K_r = K_r[:, :N, :]                              # keep 2N→N real rows
        K_i = K_i[:, :N, :]
        K    = torch.complex(K_r, K_i)                   # [B,N,W] ; W=N
        # Apply Hermitian enforcement
        # K = self._enforce_hermitian(K)

        # ── Gram + diagonal loading (same as original) ───────────
        Rz = gram_diagonal_overload(K, eps=1.0, batch_size=B)

        # Differentiable subspace method
        out = self._diff_method(Rz, self.M, B)
        if isinstance(out, tuple):
            doa_pred, doa_all, roots = out
        else:                           # ESPRIT
            doa_pred, doa_all, roots = out, None, None
        return doa_pred, doa_all, roots, Rz


# -----------------------------------------------------------------------------

if ModelRegistry is not None:  # pragma: no cover – optional integration
    ModelRegistry.register("deep_root_music")(DeepRootMUSIC)
    ModelRegistry.register("subspace_net")(SubspaceNet)
    ModelRegistry.register("subspace_net_esprit")(SubspaceNetEsprit)
    ModelRegistry.register("subspace_unet")(SubspaceUNet)
    ModelRegistry.register("deep_augmented_music")(DeepAugmentedMUSIC)
    ModelRegistry.register("deep_cnn_subspace")(DeepCNN) 