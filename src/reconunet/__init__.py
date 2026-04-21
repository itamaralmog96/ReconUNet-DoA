"""ReconUNet — deep eigen-covariance reconstruction for robust DOA estimation.

This is the reference implementation from Almog & Weiss's MSc thesis
(``Thesis_DOA-01.pdf``).  The package is organized around five subsystems:

- ``reconunet.signalgen``  — analytic array-signal generation (ULA / URA /
  triangular arrays, mutual coupling, gain/phase errors).
- ``reconunet.data``       — scene-manifest, renderer, and PyTorch datasets.
- ``reconunet.models``     — classic (MUSIC, ESPRIT, Root-MUSIC, MVDR) and
  deep-learning (EVDUNet, classification nets, ViT) estimators, plus
  third-party submodule adapters (SubspaceNet, DOA-ViT).
- ``reconunet.training``   — unified training loops for all three DL models.
- ``reconunet.evaluation`` — Monte-Carlo harnesses, CRLB comparisons, and
  publication-ready plotting.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("reconunet")
except PackageNotFoundError:  # when running from a source checkout without install
    __version__ = "0.1.0.dev0"

__all__ = ["__version__"]
