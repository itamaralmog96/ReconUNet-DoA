"""Synthetic torch verification for the paper-faithful composite loss.

Goals:
  1. Build a mixed-K batch (n_sources = [1, 2, 3, 4]) with a known clean
     covariance per sample, exactly as the renderer would produce.
  2. Drive _compute_loss_native end-to-end and check:
       a. All four terms (L_eig, L_proj, L_dom, L_rec) appear in metrics.
       b. With pred == truth the total loss is ~0.
       c. With perturbed pred the loss is positive and finite, gradients
          flow through K_recon / eigvals / eigvecs.
       d. The per-sample K mask matters: zeroing batch["n_sources"] should
          collapse L_proj to ~0 (no signal-subspace columns selected).
"""
from __future__ import annotations

import sys
import math
import torch

sys.path.insert(0, "/sessions/tender-nice-brahmagupta/mnt/All_DOA_nets/ReconUNet/src")

from reconunet.cli.train import _compute_loss_native


def make_clean_cov(M: int, K: int, device="cpu", dtype=torch.complex64) -> torch.Tensor:
    """Build R_clean = A A^H for K random angles in (-60°, 60°) on a ULA."""
    angles_deg = (torch.rand(K) - 0.5) * 120.0       # uniform in (-60, 60)
    angles_rad = angles_deg * math.pi / 180.0
    n = torch.arange(M).unsqueeze(1)                 # [M, 1]
    phase = -1j * math.pi * n * torch.sin(angles_rad).unsqueeze(0)  # [M, K]
    A = torch.exp(phase).to(dtype)                   # element_spacing_lambda=0.5
    return (A @ A.conj().transpose(-2, -1))          # [M, M]


def main() -> None:
    torch.manual_seed(20260425)
    M, B = 8, 4
    n_sources = torch.tensor([1, 2, 3, 4], dtype=torch.long)

    # ---- Build clean targets ----
    K_clean = torch.stack([make_clean_cov(M, int(k)) for k in n_sources], dim=0)

    # =========================================================================
    # Test 1: pred == truth  ->  loss ~ 0
    # =========================================================================
    with torch.no_grad():
        w_true, V_true = torch.linalg.eigh(K_clean)
        w_true = torch.flip(w_true, dims=[-1])
        V_true = torch.flip(V_true, dims=[-1])

    # Make leaf tensors so gradients can flow.
    K_recon_perfect = K_clean.clone().to(torch.complex64).requires_grad_(True)
    eigvals_pred_perfect = w_true.clone().to(torch.float32).requires_grad_(True)
    eigvecs_pred_perfect = V_true.clone().to(torch.complex64).requires_grad_(True)

    out = {
        "K_recon": K_recon_perfect,
        "eigvals": eigvals_pred_perfect,
        "eigvecs": eigvecs_pred_perfect,
    }
    batch = {
        "covariance_clean": K_clean,
        "n_sources": n_sources,
    }
    loss_cfg = {
        "eigval_weight": 1.0,
        "proj_weight": 1.0,
        "dom_weight": 1.0,
        "reconstruction_weight": 1.0,
    }
    total, metrics = _compute_loss_native(out, batch, loss_cfg)
    print("=== Test 1: prediction == truth ===")
    for k, v in sorted(metrics.items()):
        print(f"  {k:20s} = {v:.6e}")
    assert metrics["loss"] < 1e-4, f"Expected ~0 loss with perfect pred, got {metrics['loss']}"
    print("  PASS: total loss ~ 0 when prediction matches truth\n")

    # =========================================================================
    # Test 2: perturbed pred  ->  positive, finite, gradients flow
    # =========================================================================
    torch.manual_seed(7)
    noise = (torch.randn_like(K_clean.real) + 1j * torch.randn_like(K_clean.real)) * 0.1
    K_recon = (K_clean + noise).clone().to(torch.complex64).requires_grad_(True)

    # Perturb eigvecs slightly and re-orthonormalise to keep them unitary.
    V_pert = V_true + 0.05 * torch.randn_like(V_true)
    Q, _ = torch.linalg.qr(V_pert)
    eigvecs_pred = Q.clone().to(torch.complex64).requires_grad_(True)
    eigvals_pred = (w_true + 0.1 * torch.randn_like(w_true)).clone().to(torch.float32).requires_grad_(True)

    out = {
        "K_recon": K_recon,
        "eigvals": eigvals_pred,
        "eigvecs": eigvecs_pred,
    }
    total, metrics = _compute_loss_native(out, batch, loss_cfg)
    print("=== Test 2: perturbed prediction ===")
    for k, v in sorted(metrics.items()):
        print(f"  {k:20s} = {v:.6e}")
    assert metrics["loss"] > 0.0 and math.isfinite(metrics["loss"]), "Loss must be finite & positive"
    for term in ("eigval", "proj", "dom", "reconstruction"):
        assert term in metrics, f"Missing loss component {term!r}"
        assert math.isfinite(metrics[term]), f"{term} not finite"
    # Each per-term magnitude in a sane range.
    assert 0.0 < metrics["dom"] < 2.0, f"L_dom out of range: {metrics['dom']}"
    assert 0.0 < metrics["proj"], f"L_proj should be > 0 with perturbation: {metrics['proj']}"
    assert 0.0 < metrics["reconstruction"], f"L_rec should be > 0: {metrics['reconstruction']}"

    # Gradient flow check.
    total.backward()
    for name, t in [("K_recon", K_recon), ("eigvals", eigvals_pred), ("eigvecs", eigvecs_pred)]:
        assert t.grad is not None, f"No gradient on {name}"
        gnorm = float(t.grad.detach().abs().sum())
        assert gnorm > 0.0, f"Zero gradient on {name}"
        print(f"  grad {name:8s} L1-norm = {gnorm:.6e}")
    print("  PASS: all four terms positive, finite, gradients flow\n")

    # =========================================================================
    # Test 3: per-sample K mask actually depends on batch["n_sources"]
    # =========================================================================
    out = {
        "K_recon": K_recon.detach().clone().requires_grad_(True),
        "eigvals": eigvals_pred.detach().clone().requires_grad_(True),
        "eigvecs": eigvecs_pred.detach().clone().requires_grad_(True),
    }
    proj_full_K = metrics["proj"]
    batch_zero = {
        "covariance_clean": K_clean,
        "n_sources": torch.zeros(B, dtype=torch.long),  # K_b = 0 -> empty V_S
    }
    total_z, metrics_z = _compute_loss_native(out, batch_zero, loss_cfg)
    print("=== Test 3: K=0 mask collapses L_proj ===")
    print(f"  proj with K={n_sources.tolist()}: {proj_full_K:.6e}")
    print(f"  proj with K=[0,0,0,0]:           {metrics_z['proj']:.6e}")
    assert metrics_z["proj"] < 1e-12, (
        f"With K=0 every sample, V_S is empty so L_proj must be ~0; "
        f"got {metrics_z['proj']}"
    )
    print("  PASS: per-sample K mask is honoured\n")

    # =========================================================================
    # Test 4: legacy eigvec_weight alias still works (back-compat)
    # =========================================================================
    legacy_cfg = {
        "eigval_weight": 1.0,
        "eigvec_weight": 1.0,        # legacy alias for L_proj
        "reconstruction_weight": 1.0,
    }
    out = {
        "K_recon": K_recon.detach().clone().requires_grad_(True),
        "eigvals": eigvals_pred.detach().clone().requires_grad_(True),
        "eigvecs": eigvecs_pred.detach().clone().requires_grad_(True),
    }
    total_l, metrics_l = _compute_loss_native(out, batch, legacy_cfg)
    print("=== Test 4: legacy eigvec_weight alias ===")
    for k, v in sorted(metrics_l.items()):
        print(f"  {k:20s} = {v:.6e}")
    assert "eigvec" in metrics_l, "Legacy alias 'eigvec' not in metrics"
    assert abs(metrics_l["eigvec"] - metrics["proj"]) < 1e-6, (
        "Legacy eigvec_weight must produce the same value as proj_weight"
    )
    print("  PASS: legacy eigvec_weight aliases correctly to L_proj\n")

    print("All composite-loss verification tests PASSED.")


if __name__ == "__main__":
    main()
