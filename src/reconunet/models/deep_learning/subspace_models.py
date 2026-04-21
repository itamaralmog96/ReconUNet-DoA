
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

def root_music(Rz: torch.Tensor, M: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable Root-MUSIC implementation (narrow-band ULA, ideal case)."""
    # --- Compute DoA estimates using Tri4Net RootMUSIC formulation ---
    element_spacing = 0.5  # wavelength units (same default as Tri4Net classic)

    doa_batches: list[torch.Tensor] = []
    doa_all_batches: list[torch.Tensor] = []

    for b in range(batch_size):
        R = Rz[b]

        # Eigendecomposition (use eigh for Hermitian matrices)
        eigenvalues, eigenvectors = torch.linalg.eigh(R)
        sort_idx = torch.argsort(eigenvalues).flip(0)  # No need for abs() since eigenvalues are real
        Un = eigenvectors[:, sort_idx][:, M:]  # noise sub-space (N x (N-M))

        # Build polynomial coefficients from noise sub-space
        F = Un @ Un.conj().T
        coeffs = sum_of_diags_torch(F)
        roots = find_roots_torch(coeffs)

        # Shift roots as in Tri4Net to account for array center
        N = R.shape[0]
        c = N / 2 if N % 2 == 0 else (N - 1) / 2
        k_d = element_spacing * 2 * np.pi
        phase = torch.tensor(1j * k_d * c, dtype=roots.dtype, device=roots.device)
        roots_shifted = roots * torch.exp(phase)

        # Helper for degree conversion (torch lacks rad2deg before v1.8)
        _rad2deg = 180.0 / np.pi

        # All roots → provisional DoAs (for spectrum analysis / debugging)
        roots_angles_all = torch.angle(roots_shifted) / k_d  # sin(theta) style quantity
        # Clamp to valid range for arccos
        roots_angles_all = torch.clamp(roots_angles_all, min=-1.0, max=1.0)
        doa_all = torch.acos(-roots_angles_all) * _rad2deg
        doa_all_batches.append(doa_all)

        # Sort by distance to unit circle and keep those inside
        roots_sorted = roots_shifted[torch.argsort(torch.abs(torch.abs(roots_shifted) - 1.0))]
        roots_inside = roots_sorted[(torch.abs(roots_sorted) - 1) < 0][:M]

        roots_angles = torch.angle(roots_inside) / k_d
        roots_angles = torch.clamp(roots_angles, min=-1.0, max=1.0)
        doa_pred = torch.acos(-roots_angles) * _rad2deg  # degrees
        doa_batches.append(doa_pred)

    return (
        torch.stack(doa_batches, dim=0),
        torch.stack(doa_all_batches, dim=0),
        roots,  # return last roots for debugging
    )


def esprit(Rz: torch.Tensor, M: int, batch_size: int) -> torch.Tensor:
    """Differentiable ESPRIT implementation (narrow-band ULA, ideal case)."""
    doa_batches = []
    for b in range(batch_size):
        R = Rz[b]
        eigenvalues, eigenvectors = torch.linalg.eigh(R)
        sort_idx = torch.argsort(eigenvalues).flip(0)  # No need for abs() since eigenvalues are real
        Us = eigenvectors[:, sort_idx][:, :M]  # signal sub-space
        Us_upper, Us_lower = Us[:-1], Us[1:]
        phi = torch.linalg.pinv(Us_upper) @ Us_lower
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
    """Generalised subspace-method network supporting Root-MUSIC or ESPRIT."""

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