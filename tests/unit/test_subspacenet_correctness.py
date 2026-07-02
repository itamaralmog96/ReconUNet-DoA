"""Correctness tests for the SubspaceNet integration (Step 1 of the 3-way
DOA comparison).

These tests lock down the things that silently broke SubspaceNet before:

1. **Angle convention** — the differentiable ``root_music`` head returns degrees
   in ``[0, 180]`` (``= 90 + broadside_deg``); the adapter must convert it to
   broadside radians, while ``esprit`` already returns broadside radians.  Both,
   run on a clean rank-K covariance, must recover the known source angles.
2. **Differentiable eigh stability** — the custom autograd ``Function``\\ s must
   (a) survive degenerate / near-zero covariances without ``LinAlgError`` and
   produce finite gradients, and (b) match finite-difference gradients
   (``gradcheck`` on a real-valued scalar loss, which sidesteps complex-autograd
   convention pitfalls while still exercising the complex backward end-to-end).
3. **Input parity** — our ``lag_stack`` reproduces the upstream paper
   ``create_autocorrelation_tensor`` (up to the upstream global-mean subtraction,
   which the paper Eq. (32) omits and which vanishes for zero-mean signals).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from reconunet.data.scene_renderer import _steering_vector, lag_stack
from reconunet.models.deep_learning.subspace_models import (
    _DiffEspritPhi,
    _DiffNoiseProjectorBatched,
    _noise_projector_batched,
    esprit,
    gram_diagonal_overload,
    root_music,
)
from reconunet.models.third_party.subspacenet_adapter import SubspaceNetAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N = 8                      # ULA elements (paper)
SPACING = 0.5              # d / lambda (half-wavelength)


def _clean_covariance(angles_deg, powers=None, noise_var=0.01, dtype=torch.complex64):
    """Ensemble covariance ``R = A diag(p) A^H + sigma^2 I`` for an ideal ULA.

    Uses the project's own steering-vector builder with no array errors, so the
    angle convention matches the renderer / ground-truth ``angles_rad``.
    """
    angles_rad = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    K = angles_rad.shape[0]
    if powers is None:
        powers = np.ones(K)
    A = _steering_vector(
        angles_rad,
        M=N,
        element_spacing_lambda=SPACING,
        gain_err=np.ones(N),
        phase_err=np.zeros(N),
        position_err=np.zeros((N, 2)),
    )                                                  # [N, K] complex128
    A = torch.from_numpy(A)
    P = torch.from_numpy(np.diag(powers).astype(np.complex128))
    R = A @ P @ A.conj().T + noise_var * torch.eye(N, dtype=torch.complex128)
    return R.to(dtype).unsqueeze(0)                    # [1, N, N]


def _hermitian_pd_from_real(theta: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """Build a Hermitian positive-definite matrix from a *real* parameter vector.

    Lets us ``gradcheck`` a real→real composition through the complex custom
    backward without tripping over complex-autograd conventions.  ``base`` adds a
    well-separated spectrum so eigenvalues stay non-degenerate.
    """
    n = base.shape[-1]
    half = theta.numel() // 2
    re = theta[:half].reshape(n, n)
    im = theta[half:].reshape(n, n)
    Ac = torch.complex(re, im)
    return Ac @ Ac.conj().T + base


# ---------------------------------------------------------------------------
# 1. Angle convention / recovery
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("angles_deg", [[-30.0, -5.0, 20.0], [0.0, 35.0], [-15.0]])
def test_root_music_recovers_angles_via_adapter_conversion(angles_deg):
    """Root-MUSIC head + adapter conversion recovers the true broadside angles."""
    K = len(angles_deg)
    R = _clean_covariance(angles_deg, powers=np.linspace(1.0, 0.6, K))

    # _angles_from_Rz runs root_music on Rz and converts [0,180]° → broadside rad.
    adapter = SubspaceNetAdapter(diff_method="root_music")
    angles_pred = adapter._angles_from_Rz(R, K)        # broadside radians [1, K]

    got = np.sort(np.rad2deg(angles_pred[0].detach().numpy()))
    exp = np.sort(np.asarray(angles_deg))
    assert np.allclose(got, exp, atol=1.0), f"root_music got {got}, expected {exp}"


@pytest.mark.parametrize("angles_deg", [[-30.0, -5.0, 20.0], [0.0, 35.0]])
def test_esprit_recovers_angles_broadside_radians(angles_deg):
    """ESPRIT head already returns broadside radians (adapter passes through)."""
    K = len(angles_deg)
    R = _clean_covariance(angles_deg, powers=np.linspace(1.0, 0.6, K))

    adapter = SubspaceNetAdapter(diff_method="esprit")
    angles_rad = adapter._angles_from_Rz(R, K)         # esprit → broadside radians

    got = np.sort(np.rad2deg(angles_rad[0].detach().numpy()))
    exp = np.sort(np.asarray(angles_deg))
    assert np.allclose(got, exp, atol=2.0), f"esprit got {got}, expected {exp}"


# ---------------------------------------------------------------------------
# 2a. Numerical stability — degenerate inputs do not crash, grads are finite
# ---------------------------------------------------------------------------

def test_noise_projector_survives_degenerate_covariance():
    """Rz ~ I (8-fold degenerate) used to crash eigh; must now be finite."""
    B = 4
    R = torch.eye(N, dtype=torch.complex64).expand(B, N, N).contiguous()
    F = _noise_projector_batched(R, 3)
    assert F.shape == (B, N, N)
    assert torch.isfinite(F.real).all() and torch.isfinite(F.imag).all()


def test_root_music_survives_tiny_magnitude_surrogate():
    """A near-zero CNN output (K^H K ~ 0) must not raise and must give finite grads."""
    B = 4
    K = (1e-7 * torch.randn(B, N, N)).to(torch.complex64)
    K.requires_grad_(True)
    Rz = gram_diagonal_overload(K, batch_size=B)       # ~ 1e-6 * I
    doa_deg, _, _ = root_music(Rz, 3, B)
    assert torch.isfinite(doa_deg).all()
    # Gradient must flow back to K without NaN/Inf despite the degenerate spectrum.
    doa_deg.sum().backward()
    assert K.grad is not None
    assert torch.isfinite(K.grad.real).all() and torch.isfinite(K.grad.imag).all()


# ---------------------------------------------------------------------------
# 2b. gradcheck — analytic backward matches finite differences
# ---------------------------------------------------------------------------

def test_gradcheck_noise_projector():
    torch.manual_seed(0)
    n, M = 5, 2
    base = torch.diag(torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0], dtype=torch.complex128))
    W = torch.randn(1, n, n, dtype=torch.complex128)
    theta = torch.randn(2 * n * n, dtype=torch.float64, requires_grad=True)

    def g(t):
        R = _hermitian_pd_from_real(t, base).unsqueeze(0)
        F = _DiffNoiseProjectorBatched.apply(R, M)
        return (W.conj() * F).real.sum()              # real scalar

    assert torch.autograd.gradcheck(g, (theta,), eps=1e-6, atol=1e-4, rtol=1e-3)


@pytest.mark.xfail(
    reason="The differentiable ESPRIT head's custom backward is approximate: it "
    "omits the intra-signal eigenvector-coupling block and uses a simplified "
    "pseudo-inverse derivative.  ESPRIT is the secondary head — its forward is "
    "correct and stable (see angle-recovery / stability tests) — but its training "
    "gradient is not finite-difference-exact.  root_music is the validated default "
    "for the paper comparison.  Fixing the ESPRIT backward is a tracked follow-up.",
    strict=True,
)
def test_gradcheck_esprit_phi():
    torch.manual_seed(1)
    n, M = 5, 2
    base = torch.diag(torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0], dtype=torch.complex128))
    theta = torch.randn(2 * n * n, dtype=torch.float64, requires_grad=True)

    def g(t):
        R = _hermitian_pd_from_real(t, base)           # [n, n] (single matrix)
        phi = _DiffEspritPhi.apply(R, M)               # [M, M] shift-invariance matrix
        return (phi.conj() * phi).real.sum()           # real scalar (|phi|_F^2)

    assert torch.autograd.gradcheck(g, (theta,), eps=1e-6, atol=1e-4, rtol=1e-3)


# ---------------------------------------------------------------------------
# 3. lag-stack input parity vs the upstream paper implementation
# ---------------------------------------------------------------------------

def test_lag_stack_matches_upstream_autocorrelation():
    """``lag_stack`` reproduces upstream ``create_autocorrelation_tensor`` for a
    zero-mean signal (the only regime where the paper Eq. (32) and the upstream
    global-mean-subtracted form coincide)."""
    try:
        from reconunet.models.third_party.subspacenet_adapter import _ensure_on_path
        _ensure_on_path()
        from src.data_handler import create_autocorrelation_tensor  # type: ignore
    except Exception as exc:  # pragma: no cover - upstream submodule not importable
        pytest.skip(f"upstream SubspaceNet submodule not importable: {exc}")

    torch.manual_seed(0)
    M, T, tau = N, 256, 8
    X = torch.randn(M, T, dtype=torch.complex128)
    X = X - X.mean()                                   # zero global mean → formulas agree

    ours = lag_stack(X.unsqueeze(0), tau=tau)[0]       # [tau, 2M, M] float32
    upstream = create_autocorrelation_tensor(X, tau).cpu().to(torch.float32)

    assert ours.shape == upstream.shape == (tau, 2 * M, M)
    assert torch.allclose(ours, upstream, atol=1e-3, rtol=1e-3)
