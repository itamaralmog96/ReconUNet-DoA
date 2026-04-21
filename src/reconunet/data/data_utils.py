"""Utility helpers for Tri4Net data processing.

This module duplicates a subset of SubspaceNet's `data_handler` API so that
training can be **independent** of the original SubspaceNet repository.

Implemented functions
---------------------
* `autocorrelation_matrix`      - compute Rx(τ) for a single lag
* `create_autocorrelation_tensor` - stack Rx(τ) for τ = 0 … (T-1)
* `create_cov_tensor`           - 3-channel (Re, Im, phase) covariance image

All functions operate on *single-sample* tensors.  Batched processing is left
to the caller/DataLoader.
"""
from __future__ import annotations

import torch
import numpy as np
import h5py

__all__ = [
    "autocorrelation_matrix",
    "create_autocorrelation_tensor", 
    "create_cov_tensor",
]


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Autocorrelation helpers ------------------------------------------------------
# -----------------------------------------------------------------------------

def autocorrelation_matrix(X: torch.Tensor, lag: int) -> torch.Tensor:
    """Compute the (biased) sample autocorrelation matrix at a given *lag*.

    Parameters
    ----------
    X : torch.Tensor, shape (N, T), **complex**64/128
        Observation/snapshots matrix.  *N* sensors, *T* snapshots.
    lag : int
        Desired lag 0 ≤ lag < T.

    Returns
    -------
    torch.Tensor, shape (2N, N), **float32**
        Real and imaginary parts stacked vertically (matching SubspaceNet).
    """
    if lag < 0 or lag >= X.shape[1]:
        raise ValueError("lag must be 0 ≤ lag < T")

    # Subtract mean ‒ improves numerical stability (optional)
    Xc = X - X.mean(dim=1, keepdim=True)

    # Vectors for the two time indices
    X1 = Xc[:, : X.shape[1] - lag]
    X2 = Xc[:, lag:]

    # Compute Rx_lag = 1/(T - lag) * Σ_{t} x(t) x(t+lag)^H
    Rx = (X1 @ X2.conj().T) / (X1.shape[1])  # [N, N] complex

    # Stack real/imag vertically to match shape expected by SubspaceNet
    Rx_cat = torch.cat((Rx.real, Rx.imag), dim=0)  # [2N, N]
    return Rx_cat.to(torch.float32)


def create_autocorrelation_tensor(X: torch.Tensor, tau: int) -> torch.Tensor:
    """Return a tensor of autocorrelation matrices for lags 0 … τ-1.

    Shape: (tau, 2N, N)
    """
    if tau <= 0:
        raise ValueError("tau must be positive")

    Rx_list = [autocorrelation_matrix(X, lag=i) for i in range(tau)]
    return torch.stack(Rx_list, dim=0)  # [tau, 2N, N]


# -----------------------------------------------------------------------------
# Covariance image -------------------------------------------------------------
# -----------------------------------------------------------------------------

def create_cov_tensor(X: torch.Tensor) -> torch.Tensor:
    """Create a 3-channel tensor [Re, Im, Phase] of the spatial covariance.

    Parameters
    ----------
    X : torch.Tensor, shape (N, T), **complex**

    Returns
    -------
    torch.Tensor, shape (N, N, 3), **float32**
    """
    # Spatial covariance (biased)
    Rx = (X @ X.conj().T) / X.shape[1]  # [N, N] complex

    phase = torch.angle(Rx)
    Rx_tensor = torch.stack((Rx.real, Rx.imag, phase), dim=2)  # [N, N, 3]
    return Rx_tensor.to(torch.float32)
