"""Batched classical DoA estimators on a covariance, in the project's broadside
sine convention (matching :func:`reconunet.data.scene_renderer._steering_vector`:
``a_m(θ) = exp(-j·2π·m·d·sinθ)``).

All estimators take a complex covariance ``R`` of shape ``[B, M, M]`` and the
source count ``K`` and return predicted broadside angles in **radians**, shape
``[B, K]`` — directly comparable to ``angles_rad`` in the manifests.

Covers the full classical lineup used by Almog & Weiss:
  * spectral:  bartlett (beamformer), mvdr (Capon), music
  * gridless:  root_music, esprit (wrappers over subspace_models), unitary_esprit
"""
from __future__ import annotations

import numpy as np
import torch

from reconunet.models.deep_learning.subspace_models import root_music as _root_music
from reconunet.models.deep_learning.subspace_models import esprit as _esprit


# ---------------------------------------------------------------------------
# Steering + peak-picking
# ---------------------------------------------------------------------------

def steering_matrix(grid_rad: torch.Tensor, M: int, d: float = 0.5) -> torch.Tensor:
    """``[M, G]`` complex steering matrix for broadside angles ``grid_rad``."""
    m = torch.arange(M, dtype=torch.float64, device=grid_rad.device).unsqueeze(1)   # [M, 1]
    phase = -2.0 * np.pi * d * m * torch.sin(grid_rad).unsqueeze(0)  # [M, G]
    return torch.exp(1j * phase)                                    # [M, G] complex128


def _peak_pick(spectrum: torch.Tensor, K: int, grid_rad: torch.Tensor) -> torch.Tensor:
    """Return the K strongest *local maxima* per row → broadside angles [B, K].

    Applies three-point parabolic refinement around each selected grid peak
    (paper §IV: "1° grid … with three-point parabolic refinement around peaks").
    Falls back to global top-K if a row has fewer than K interior local maxima.
    """
    B, G = spectrum.shape
    interior = spectrum[:, 1:-1]
    is_max = (interior > spectrum[:, :-2]) & (interior >= spectrum[:, 2:])     # [B, G-2]
    mask = torch.zeros_like(spectrum, dtype=torch.bool)
    mask[:, 1:-1] = is_max
    neg = torch.finfo(spectrum.dtype).min
    masked = torch.where(mask, spectrum, torch.full_like(spectrum, neg))
    n_max = mask.sum(dim=1)
    short = n_max < K
    if short.any():
        masked[short] = spectrum[short]
    idx = torch.topk(masked, k=K, dim=-1).indices                            # [B, K]

    # three-point parabolic refinement around each peak index (in grid units)
    step = grid_rad[1] - grid_rad[0]
    ic = idx.clamp(1, G - 2)
    rows = torch.arange(B, device=spectrum.device).unsqueeze(1)
    ym1 = spectrum[rows, ic - 1]; y0 = spectrum[rows, ic]; yp1 = spectrum[rows, ic + 1]
    denom = (ym1 - 2.0 * y0 + yp1)
    delta = torch.where(denom.abs() > 1e-12, 0.5 * (ym1 - yp1) / denom, torch.zeros_like(denom))
    delta = delta.clamp(-0.5, 0.5)                                           # stay within the cell
    ang = grid_rad[ic] + delta * step                                        # [B, K] refined
    return torch.sort(ang, dim=-1).values


# ---------------------------------------------------------------------------
# Spectral estimators
# ---------------------------------------------------------------------------

def _default_grid(device, lo_deg=-60.0, hi_deg=60.0, step_deg=1.0):
    """Paper §IV scan grid: 1° on [30°,150°] array-axis = [-60°,60°] broadside."""
    return torch.deg2rad(torch.arange(lo_deg, hi_deg + step_deg / 2, step_deg,
                                      dtype=torch.float64, device=device))


def bartlett(R: torch.Tensor, K: int, M: int, d: float = 0.5, grid_rad=None) -> torch.Tensor:
    """Conventional (Bartlett) beamformer: P(θ)=aᴴ R a."""
    dev = R.device
    R = R.to(torch.complex128)          # fp64 linear algebra (robustness/precision)
    grid = _default_grid(dev) if grid_rad is None else grid_rad
    A = steering_matrix(grid, M, d).to(dev)                                  # [M, G]
    RA = torch.einsum("bmn,ng->bmg", R, A)                                   # [B, M, G]
    P = torch.einsum("mg,bmg->bg", A.conj(), RA).real                        # [B, G]
    return _peak_pick(P, K, grid)


def mvdr(R: torch.Tensor, K: int, M: int, d: float = 0.5, grid_rad=None) -> torch.Tensor:
    """MVDR / Capon: P(θ)=1/(aᴴ R⁻¹ a)."""
    dev = R.device
    R = R.to(torch.complex128)          # fp64 inverse (robustness/precision)
    grid = _default_grid(dev) if grid_rad is None else grid_rad
    A = steering_matrix(grid, M, d).to(dev)
    eye = torch.eye(M, dtype=R.dtype, device=dev).unsqueeze(0)
    load = 1e-3 * R.diagonal(dim1=-2, dim2=-1).real.mean(-1)[:, None, None]
    Rinv = torch.linalg.inv(R + load * eye)                                  # [B, M, M]
    RiA = torch.einsum("bmn,ng->bmg", Rinv, A)
    denom = torch.einsum("mg,bmg->bg", A.conj(), RiA).real.clamp_min(1e-12)
    return _peak_pick(1.0 / denom, K, grid)


def music(R: torch.Tensor, K: int, M: int, d: float = 0.5, grid_rad=None) -> torch.Tensor:
    """Spectral MUSIC: P(θ)=1/‖Enᴴ a‖²."""
    dev = R.device
    # fp64 CPU eigh — LAPACK is faster and far more robust than cuSOLVER for
    # these small (8×8) Hermitian matrices (see _robust_hermitian_eigh notes);
    # network-reconstructed covariances can have clustered spectra that trip
    # complex64 cuSOLVER.
    R64 = R.detach().to("cpu", torch.complex128)
    grid = _default_grid(dev) if grid_rad is None else grid_rad
    A = steering_matrix(grid, M, d).to(dev)
    # ascending eigenvalues → noise subspace = first M-K columns
    w, V = torch.linalg.eigh(R64)
    En = V[:, :, : M - K].to(dev)                                            # [B, M, M-K]
    EA = torch.einsum("bmk,mg->bkg", En.conj(), A)                           # [B, M-K, G]
    denom = (EA.abs() ** 2).sum(dim=1).clamp_min(1e-12)                      # [B, G]
    return _peak_pick(1.0 / denom, K, grid)


# ---------------------------------------------------------------------------
# Gridless estimators
# ---------------------------------------------------------------------------

def root_music(R: torch.Tensor, K: int) -> torch.Tensor:
    """Root-MUSIC wrapper → broadside radians [B, K] (subspace_models convention)."""
    deg = _root_music(R, K, R.shape[0])[0]                                   # [B, *] in [0,180]
    return torch.deg2rad(deg[:, :K] - 90.0)


def esprit(R: torch.Tensor, K: int) -> torch.Tensor:
    """ESPRIT wrapper → broadside radians [B, K]."""
    return _esprit(R, K, R.shape[0])[:, :K]


def unitary_esprit(R: torch.Tensor, K: int, M: int, d: float = 0.5) -> torch.Tensor:
    """Real-valued Unitary-ESPRIT (LS variant) with forward-backward averaging.

    Uses the sparse unitary transform that maps a centro-Hermitian matrix to a
    real one (Haardt & Nossek, 1995), then solves the real rotational-invariance
    equation on the max-overlap subarrays.  Returns broadside radians [B, K].
    """
    dev = R.device
    B = R.shape[0]
    Rc = R.detach().to(torch.complex128).cpu()
    Pi = torch.flip(torch.eye(M, dtype=torch.complex128), dims=[0])          # exchange matrix
    # forward-backward smoothing → centro-Hermitian
    R_fb = 0.5 * (Rc + Pi @ Rc.conj() @ Pi)

    def Q(n):
        """Sparse unitary Q_n mapping centro-Hermitian → real (n even/odd)."""
        half = n // 2
        I = torch.eye(half, dtype=torch.complex128)
        J = torch.flip(I, dims=[0])
        j = 1j
        if n % 2 == 0:
            top = torch.cat([I, j * I], dim=1)
            bot = torch.cat([J, -j * J], dim=1)
            Qn = torch.cat([top, bot], dim=0) / np.sqrt(2.0)
        else:
            z = torch.zeros(half, 1, dtype=torch.complex128)
            top = torch.cat([I, z, j * I], dim=1)
            mid = torch.cat([torch.zeros(1, half, dtype=torch.complex128),
                             torch.tensor([[np.sqrt(2.0)]], dtype=torch.complex128),
                             torch.zeros(1, half, dtype=torch.complex128)], dim=1)
            bot = torch.cat([J, z, -j * J], dim=1)
            Qn = torch.cat([top, mid, bot], dim=0) / np.sqrt(2.0)
        return Qn

    Qm = Q(M)
    Qm1 = Q(M - 1)
    # real transformed covariance  C = Re( Qmᴴ R_fb Qm )  (imag ≈ 0 by construction)
    C = (Qm.conj().mT @ R_fb @ Qm).real                                      # [B, M, M] real
    # selection matrices on the two max-overlap subarrays
    J2 = torch.zeros(M - 1, M, dtype=torch.complex128)
    for i in range(M - 1):
        J2[i, i + 1] = 1.0
    K1 = (Qm1.conj().mT @ J2 @ Qm).real                                      # [M-1, M]
    K2 = (Qm1.conj().mT @ J2 @ Qm).imag

    out = torch.full((B, K), float("nan"), dtype=torch.float64)
    for b in range(B):
        w, Es = torch.linalg.eigh(C[b])                                      # ascending, real
        Es = Es[:, M - K:]                                                   # [M, K] signal (real)
        # real LS:  (K1 Es) Υ = (K2 Es)
        lhs = K1 @ Es                                                        # [M-1, K] real
        rhs = K2 @ Es
        Ups = torch.linalg.pinv(lhs) @ rhs                                   # [K, K] real
        ev = torch.linalg.eigvals(Ups)                                       # complex
        mu = 2.0 * torch.atan(ev.real)                                       # spatial freq (rad)
        # renderer phase per element is -2π·d·sinθ, so sinθ = -μ/(2π·d).
        sin_t = (-mu / (2.0 * np.pi * d)).clamp(-1.0, 1.0)
        ang = torch.arcsin(sin_t)                                            # broadside rad
        ang = torch.sort(ang)[0]
        out[b, : ang.numel()] = ang[:K]
    return out.to(dev)


ESTIMATORS = {
    "Bartlett":       lambda R, K, M: bartlett(R, K, M),
    "MVDR":           lambda R, K, M: mvdr(R, K, M),
    "MUSIC":          lambda R, K, M: music(R, K, M),
    "Root-MUSIC":     lambda R, K, M: root_music(R, K),
    "ESPRIT":         lambda R, K, M: esprit(R, K),
    "Unitary-ESPRIT": lambda R, K, M: unitary_esprit(R, K, M),
}
