"""Loss functions for Tri4Net deep-learning training.

Currently provides multiple RMSPE (Root-Mean-Square Periodic Error) variants
for different angle ranges commonly used in DOA estimation.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from itertools import permutations

__all__ = ["RMSPELoss", "RMSPELoss_0_180", "RMSPELoss_0_360"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _permute_prediction(prediction: torch.Tensor) -> torch.Tensor:
    """Generate every permutation of the *prediction* vector (along dim-0)."""
    idx_perms = list(permutations(range(prediction.shape[0]), prediction.shape[0]))
    return torch.stack(
        [prediction.index_select(0, torch.tensor(p, device=prediction.device)) for p in idx_perms],
        dim=0,
    )


class RMSPELoss(nn.Module):
    """Root-Mean-Square Periodic Error between angle predictions & targets (radians).

    For angles in range [-π/2, π/2] (i.e., -90° to 90°).
    
    The metric wraps each angle difference into the range [-π/2, π/2] and then
    chooses the **minimum** error over all permutations of the *M* predicted
    angles, ensuring label-order invariance.
    """

    def __init__(self):  # noqa: D401 – simple initializer
        super().__init__()

    # ------------------------------------------------------------------
    def forward(self, doa_pred: torch.Tensor, doa_true: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Compute batch RMSPE.

        Parameters
        ----------
        doa_pred : (B, M) float tensor - predictions in **radians** [-π/2, π/2]
        doa_true : (B, M) float tensor - ground-truth angles in **radians** [-π/2, π/2]
        """
        batch_errors = []
        for b in range(doa_pred.shape[0]):
            preds = doa_pred[b].to(DEVICE)
            targets = doa_true[b].to(DEVICE)

            # All permutations of predictions (order-invariance)
            permuted = _permute_prediction(preds)

            # Error modulo π (wrap into [-π/2, π/2])
            err = (((permuted - targets) + (np.pi / 2)) % np.pi) - (np.pi / 2)  # [P, M]

            # RMS over sensors (M)
            rms = (1.0 / np.sqrt(targets.numel())) * torch.linalg.norm(err, dim=1)  # [P]

            batch_errors.append(torch.min(rms))

        return torch.sum(torch.stack(batch_errors))


class RMSPELoss_0_180(nn.Module):
    """Root-Mean-Square Periodic Error for angles in range [0, π] (0° to 180°).

    For many DOA estimation scenarios where angles are measured from 0° to 180°,
    typically without wraparound (i.e., 0° and 180° are treated as distinct directions).
    If wraparound behavior is needed, set wraparound=True.
    """

    def __init__(self, wraparound: bool = False):
        """Initialize the loss function.
        
        Parameters
        ----------
        wraparound : bool
            If True, treats 0° and 180° as potentially equivalent (circular wraparound).
            If False, treats them as distinct directions (default behavior).
        """
        super().__init__()
        self.wraparound = wraparound

    def forward(self, doa_pred: torch.Tensor, doa_true: torch.Tensor) -> torch.Tensor:
        """Compute batch RMSPE.

        Parameters
        ----------
        doa_pred : (B, M) float tensor - predictions in **radians** [0, π]
        doa_true : (B, M) float tensor - ground-truth angles in **radians** [0, π]
        """
        batch_errors = []
        for b in range(doa_pred.shape[0]):
            preds = doa_pred[b].to(DEVICE)
            targets = doa_true[b].to(DEVICE)

            # All permutations of predictions (order-invariance)
            permuted = _permute_prediction(preds)

            if self.wraparound:
                # Handle wraparound: consider both direct difference and wraparound difference
                diff = permuted - targets  # [P, M]
                # For [0, π] range, wraparound distance is min(|diff|, π - |diff|)
                abs_diff = torch.abs(diff)
                wraparound_diff = np.pi - abs_diff
                err = torch.minimum(abs_diff, wraparound_diff) * torch.sign(diff)  # [P, M]
            else:
                # No wraparound: simple difference
                err = permuted - targets  # [P, M]

            # RMS over sensors (M)
            rms = (1.0 / np.sqrt(targets.numel())) * torch.linalg.norm(err, dim=1)  # [P]

            batch_errors.append(torch.min(rms))

        return torch.sum(torch.stack(batch_errors))


class RMSPELoss_0_360(nn.Module):
    """Root-Mean-Square Periodic Error for angles in range [0, 2π] (0° to 360°).

    For full 360° DOA estimation where 0° and 360° are equivalent (wraparound).
    This version always considers the circular nature of the angle space.
    """

    def __init__(self):
        super().__init__()

    def forward(self, doa_pred: torch.Tensor, doa_true: torch.Tensor) -> torch.Tensor:
        """Compute batch RMSPE.

        Parameters
        ----------
        doa_pred : (B, M) float tensor - predictions in **radians** [0, 2π]
        doa_true : (B, M) float tensor - ground-truth angles in **radians** [0, 2π]
        """
        batch_errors = []
        for b in range(doa_pred.shape[0]):
            preds = doa_pred[b].to(DEVICE)
            targets = doa_true[b].to(DEVICE)

            # All permutations of predictions (order-invariance)
            permuted = _permute_prediction(preds)

            # Circular difference for [0, 2π] range
            diff = permuted - targets  # [P, M]
            
            # Wrap differences to [-π, π] to find shortest circular distance
            err = (diff + np.pi) % (2 * np.pi) - np.pi  # [P, M]

            # RMS over sensors (M)
            rms = (1.0 / np.sqrt(targets.numel())) * torch.linalg.norm(err, dim=1)  # [P]

            batch_errors.append(torch.min(rms))

        return torch.sum(torch.stack(batch_errors)) 