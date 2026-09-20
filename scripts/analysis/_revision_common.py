"""Shared helpers for the 2026-09-20 revision evaluation scripts (sweeps, FBSS,
figures, MUSIC verification).  Loads the released checkpoints once and exposes
thin prediction functions with a common contract: inputs are complex snapshots
``[g, M, T]``, outputs broadside radians ``[g, K]``."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from reconunet.data.scene_renderer import lag_stack
from reconunet.models.deep_learning.subspace_models import root_music as _rm
from reconunet.models.third_party.damusic_adapter import DAMUSICEnsemble
from reconunet.models.third_party.subspacenet_adapter import SubspaceNetAdapter

REPO = Path(__file__).resolve().parents[2]
CK = {"reconunet": REPO / "experiments/runs/reconunet_paper/checkpoints/best.pt",
      "subspacenet": REPO / "experiments/runs/subspacenet_paper/checkpoints/best.pt",
      "damusic_dir": REPO / "experiments/runs/damusic_paper"}


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def scm(snaps: torch.Tensor) -> torch.Tensor:
    return (snaps @ snaps.conj().transpose(-1, -2)) / snaps.shape[-1]


def root_music_rad(R: torch.Tensor, K: int) -> np.ndarray:
    """Root-MUSIC on a batch of covariances -> broadside radians [g, K]."""
    return np.deg2rad(_rm(R.detach().cpu(), K, R.shape[0])[0].numpy() - 90.0)[:, :K]


def sq_err_deg2(pred_rad: np.ndarray, true_rad: np.ndarray) -> np.ndarray:
    """Per-source squared errors in deg² with sorted pairing (paper eq. 31), no wrap."""
    p = np.sort(pred_rad, axis=-1)[:, : true_rad.shape[1]]
    t = np.sort(true_rad, axis=-1)
    return np.rad2deg(p - t) ** 2


class Models:
    """ReconUNet (+Root-MUSIC), SubspaceNet (+Root-MUSIC head), DA-MUSIC ensemble."""

    def __init__(self, dev, tau: int = 8, reconunet=CK["reconunet"], subspacenet=CK["subspacenet"],
                 damusic_dir=CK["damusic_dir"], with_damusic: bool = True):
        from reconunet.cli.evaluate import _NativeEVDUNetAdapter
        self.dev, self.tau = dev, tau
        self.rn_ad = _NativeEVDUNetAdapter("reconunet.models.deep_learning.EVDUNet.EVDCovarianceReconstructionUNet")
        ck = torch.load(reconunet, map_location="cpu", weights_only=False)
        init = dict(ck["cfg"]["model"]["init"]) if isinstance(ck, dict) and "cfg" in ck else \
            {"M": 8, "tau": tau, "activation_type": "anti_rectifier", "use_dropout": True}
        self.rn = self.rn_ad.build_model(init); self.rn_ad.load_checkpoint(self.rn, str(reconunet)); self.rn.to(dev).eval()
        self.rn_epoch = int(ck.get("epoch", -1)) if isinstance(ck, dict) else -1
        self.sn_ad = SubspaceNetAdapter(M=4, tau=tau, diff_method="root_music")
        self.sn = self.sn_ad.build_model({"M": 4, "tau": tau, "diff_method": "root_music"})
        self.sn_ad.load_checkpoint(self.sn, str(subspacenet)); self.sn.to(dev).eval()
        self.dm = DAMUSICEnsemble.from_run_dir(damusic_dir, device=dev) if with_damusic else None

    @torch.no_grad()
    def reconunet_cov(self, snaps: torch.Tensor) -> torch.Tensor:
        """Reconstructed covariance R_hat [g, M, M] (on device)."""
        _, _, R_hat = self.rn(lag_stack(snaps, tau=self.tau).to(self.dev))
        return R_hat

    @torch.no_grad()
    def reconunet(self, snaps: torch.Tensor, K: int) -> np.ndarray:
        return root_music_rad(self.reconunet_cov(snaps), K)

    @torch.no_grad()
    def subspacenet(self, snaps: torch.Tensor, K: int) -> np.ndarray:
        g = snaps.shape[0]
        out = self.sn_ad.forward(self.sn, lag_stack(snaps, tau=self.tau).to(self.dev),
                                 meta={"tau": self.tau, "n_sources": torch.full((g,), K)})
        return out.angles_pred.cpu().numpy()[:, :K]

    @torch.no_grad()
    def damusic(self, snaps: torch.Tensor, K: int):
        if self.dm is None or K not in self.dm.ks:
            return None
        g = snaps.shape[0]
        return self.dm.predict(snaps.to(self.dev), torch.full((g,), K)).cpu().numpy()[:, :K]

    @staticmethod
    def root_music(snaps: torch.Tensor, K: int) -> np.ndarray:
        return root_music_rad(scm(snaps), K)
