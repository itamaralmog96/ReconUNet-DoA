"""Compact scene manifest — the storage replacement for the old 4.3 GB H5 dataset.

Motivation
----------
The legacy dataset persisted ~2 M fully-rendered samples (raw snapshots,
covariance matrices, eigen-decompositions) into HDF5.  With 8-antenna ULAs,
512 snapshots, and complex64 that amounts to roughly 32 KB per sample — so the
set weighed in at ≈ 60 GB on disk, of which only ~200 MB is *information*.
Everything else is deterministic re-rendering of a handful of scalar
parameters through the signal model.

The :class:`SceneManifest` class stores only the parameters and relies on a
matching :class:`~reconunet.data.scene_renderer.SceneRenderer` to reconstruct
the signals on demand.  A single sample occupies **96 bytes**:

    ┌──────────────────┬────────┬──────┬──────────────────────────────┐
    │ field            │ dtype  │ bytes│ meaning                      │
    ├──────────────────┼────────┼──────┼──────────────────────────────┤
    │ seed             │ uint32 │   4  │ per-sample noise seed        │
    │ n_sources        │ uint8  │   1  │ K ∈ [1, K_max]               │
    │ _pad             │ uint8  │   3  │ align to 4-byte boundary     │
    │ angles_deg       │ float32│  32  │ up to K_max=8 angles, padded │
    │ snr_db           │ float32│   4  │ per-sample SNR               │
    │ array_type       │ uint8  │   1  │ enum ULA / URA / TRIANGULAR  │
    │ modulation       │ uint8  │   1  │ enum narrowband / wideband   │
    │ gain_err_dB      │ float16│   2  │ array gain-error strength    │
    │ phase_err_deg    │ float16│   2  │ array phase-error strength   │
    │ mutual_coupling  │ float16│   2  │ mutual-coupling coefficient  │
    │ position_err_pct │ float16│   2  │ fractional position jitter   │
    │ scene_id         │ uint64 │   8  │ cross-split trace id         │
    │ _pad2            │ uint8  │  34  │ reserved for future fields   │
    ├──────────────────┼────────┼──────┼──────────────────────────────┤
    │ TOTAL                             96                              │
    └──────────────────┴────────┴──────┴──────────────────────────────┘

For 2 M samples that is **192 MB** — a 320× reduction versus raw rendering.

Shared, per-dataset metadata (frequency, M, T, τ, sampling strategy, etc.)
lives in a separate YAML file next to the manifest so one schema change does
not force a manifest rewrite.  Together, one *(manifest.parquet, meta.yaml)*
pair fully specifies a reproducible dataset.

Design notes
------------
* We use Parquet instead of ``numpy.save`` so rows are cheaply queryable
  (e.g. "give me all samples with SNR < −10 dB") without a full load.
* A fixed ``K_max`` of 8 simplifies the schema; unused angle slots are NaN.
* ``array_type`` and ``modulation`` are ``uint8`` enums rather than strings so
  rows stay fixed-width.
"""

from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

K_MAX = 8              # maximum sources per scene (pad with NaN)
ANGLE_FILL = np.float32("nan")

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ArrayType(enum.IntEnum):
    """Array geometry selector, mirrored 1-to-1 into the renderer."""

    ULA = 0
    URA = 1
    TRIANGULAR = 2


class ModulationType(enum.IntEnum):
    """Signal modulation selector."""

    NARROWBAND = 0
    WIDEBAND = 1
    BPSK = 2
    QPSK = 3


# ---------------------------------------------------------------------------
# Per-sample record
# ---------------------------------------------------------------------------


_ROW_DTYPE = np.dtype(
    [
        ("seed", np.uint32),
        ("n_sources", np.uint8),
        ("_pad0", np.uint8, (3,)),
        ("angles_deg", np.float32, (K_MAX,)),
        ("snr_db", np.float32),
        ("array_type", np.uint8),
        ("modulation", np.uint8),
        ("gain_err_dB", np.float16),
        ("phase_err_deg", np.float16),
        ("mutual_coupling", np.float16),
        ("position_err_pct", np.float16),
        ("scene_id", np.uint64),
        ("_pad1", np.uint8, (34,)),
    ],
    align=False,
)
assert _ROW_DTYPE.itemsize == 96, f"row size drifted to {_ROW_DTYPE.itemsize}B"


@dataclass(frozen=True)
class Scene:
    """Parsed, type-safe view of one row of the manifest.

    Prefer :meth:`SceneManifest.iter_rows` in hot loops — instantiating this
    dataclass allocates and is ~5× slower than reading the structured array
    directly.
    """

    seed: int
    n_sources: int
    angles_deg: np.ndarray       # shape (K_MAX,), trailing slots are NaN
    snr_db: float
    array_type: ArrayType
    modulation: ModulationType
    gain_err_dB: float
    phase_err_deg: float
    mutual_coupling: float
    position_err_pct: float
    scene_id: int

    @property
    def angles_valid(self) -> np.ndarray:
        """The first ``n_sources`` angles, as a view — guaranteed non-NaN."""
        return self.angles_deg[: self.n_sources]


# ---------------------------------------------------------------------------
# Shared per-dataset config
# ---------------------------------------------------------------------------


@dataclass
class ManifestMeta:
    """Static array / signal parameters that are identical for every sample.

    These are **not** stored per-row to save space and to force discipline:
    changing ``fs_Hz`` requires a new manifest, not a silent row rewrite.
    """

    # --- array geometry --------------------------------------------------
    M: int = 8                       # number of elements
    element_spacing_lambda: float = 0.5
    array_type: str = "ULA"          # informational; per-row ``array_type`` overrides

    # --- signal model ----------------------------------------------------
    T: int = 512                     # snapshots per sample
    fs_Hz: float = 2.45e9            # carrier / sampling frequency
    speed_of_light: float = 2.998e8
    tau: int = 8                     # lag-stack depth (SubspaceNet / ReconUNet)

    # --- sampling strategy ----------------------------------------------
    K_max: int = K_MAX
    angle_range_deg: Tuple[float, float] = (-60.0, 60.0)
    snr_range_db: Tuple[float, float] = (-10.0, 20.0)

    # --- provenance ------------------------------------------------------
    version: str = "1.0"
    created_utc: str = ""            # ISO 8601 timestamp, filled on save
    notes: str = ""

    def to_yaml(self, path: PathLike) -> None:
        payload: Dict[str, Any] = {
            "M": self.M,
            "element_spacing_lambda": self.element_spacing_lambda,
            "array_type": self.array_type,
            "T": self.T,
            "fs_Hz": self.fs_Hz,
            "speed_of_light": self.speed_of_light,
            "tau": self.tau,
            "K_max": self.K_max,
            "angle_range_deg": list(self.angle_range_deg),
            "snr_range_db": list(self.snr_range_db),
            "version": self.version,
            "created_utc": self.created_utc,
            "notes": self.notes,
        }
        Path(path).write_text(yaml.safe_dump(payload, sort_keys=False))

    @classmethod
    def from_yaml(cls, path: PathLike) -> "ManifestMeta":
        raw = yaml.safe_load(Path(path).read_text())
        raw["angle_range_deg"] = tuple(raw.get("angle_range_deg", (-60.0, 60.0)))
        raw["snr_range_db"] = tuple(raw.get("snr_range_db", (-10.0, 20.0)))
        return cls(**raw)


# ---------------------------------------------------------------------------
# Manifest container
# ---------------------------------------------------------------------------


class SceneManifest:
    """Append-only, memory-mappable table of scene parameters.

    The on-disk format is a structured ``.npy`` (fixed 96-byte rows) plus a
    sibling ``.yaml`` for :class:`ManifestMeta`.  ``.npy`` is preferred over
    Parquet here because random access by integer index is O(1) and we can
    ``np.memmap`` for zero-copy reads from DataLoader workers.

    Typical usage::

        man = SceneManifest.create(
            meta=ManifestMeta(M=8, T=512, tau=8),
            size=2_000_000,
        )
        rng = np.random.default_rng(42)
        for i in range(len(man)):
            man[i] = random_scene(rng, man.meta)
        man.save("data/scenes/train.npy")
    """

    def __init__(self, rows: np.ndarray, meta: ManifestMeta):
        if rows.dtype != _ROW_DTYPE:
            raise TypeError(
                f"SceneManifest rows must use the canonical dtype; got {rows.dtype}"
            )
        self._rows = rows
        self.meta = meta

    # --- factories ----------------------------------------------------------

    @classmethod
    def create(cls, meta: ManifestMeta, size: int) -> "SceneManifest":
        """Allocate an empty manifest of ``size`` rows."""
        rows = np.zeros(size, dtype=_ROW_DTYPE)
        rows["angles_deg"] = ANGLE_FILL
        return cls(rows, meta)

    @classmethod
    def load(cls, npy_path: PathLike, yaml_path: Optional[PathLike] = None) -> "SceneManifest":
        npy_path = Path(npy_path)
        yaml_path = Path(yaml_path) if yaml_path else npy_path.with_suffix(".yaml")
        rows = np.load(npy_path, mmap_mode="r")
        meta = ManifestMeta.from_yaml(yaml_path)
        return cls(np.asarray(rows), meta)

    # --- persistence --------------------------------------------------------

    def save(self, npy_path: PathLike, yaml_path: Optional[PathLike] = None) -> None:
        npy_path = Path(npy_path)
        yaml_path = Path(yaml_path) if yaml_path else npy_path.with_suffix(".yaml")
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, self._rows)
        self.meta.to_yaml(yaml_path)

    # --- sequence protocol --------------------------------------------------

    def __len__(self) -> int:
        return int(self._rows.shape[0])

    def __getitem__(self, index: Union[int, slice, np.ndarray]) -> Scene:
        if isinstance(index, (slice, np.ndarray, list)):
            raise TypeError("SceneManifest supports scalar indexing only; use iter_rows()")
        r = self._rows[int(index)]
        return Scene(
            seed=int(r["seed"]),
            n_sources=int(r["n_sources"]),
            angles_deg=np.asarray(r["angles_deg"], dtype=np.float32).copy(),
            snr_db=float(r["snr_db"]),
            array_type=ArrayType(int(r["array_type"])),
            modulation=ModulationType(int(r["modulation"])),
            gain_err_dB=float(r["gain_err_dB"]),
            phase_err_deg=float(r["phase_err_deg"]),
            mutual_coupling=float(r["mutual_coupling"]),
            position_err_pct=float(r["position_err_pct"]),
            scene_id=int(r["scene_id"]),
        )

    def __setitem__(self, index: int, scene: Scene) -> None:
        r = self._rows[int(index)]
        r["seed"] = np.uint32(scene.seed)
        r["n_sources"] = np.uint8(scene.n_sources)
        angles = np.full((K_MAX,), ANGLE_FILL, dtype=np.float32)
        angles[: scene.n_sources] = scene.angles_deg[: scene.n_sources]
        r["angles_deg"] = angles
        r["snr_db"] = np.float32(scene.snr_db)
        r["array_type"] = np.uint8(int(scene.array_type))
        r["modulation"] = np.uint8(int(scene.modulation))
        r["gain_err_dB"] = np.float16(scene.gain_err_dB)
        r["phase_err_deg"] = np.float16(scene.phase_err_deg)
        r["mutual_coupling"] = np.float16(scene.mutual_coupling)
        r["position_err_pct"] = np.float16(scene.position_err_pct)
        r["scene_id"] = np.uint64(scene.scene_id)

    def iter_rows(self) -> Iterator[np.void]:
        """Iterate raw structured rows — fastest option for hot loops."""
        return iter(self._rows)

    # --- bulk helpers -------------------------------------------------------

    @property
    def raw(self) -> np.ndarray:
        """The underlying structured array, for vectorized filtering.

        Example::

            low_snr_mask = man.raw["snr_db"] < -5.0
            low_snr_indices = np.where(low_snr_mask)[0]
        """
        return self._rows

    def fingerprint(self) -> str:
        """Stable hash over the entire manifest — useful as a cache key."""
        h = hashlib.sha256()
        h.update(self._rows.tobytes())
        h.update(json.dumps(self.meta.__dict__, sort_keys=True, default=str).encode())
        return h.hexdigest()

    # --- convenience constructors for experiments ---------------------------

    @classmethod
    def random(
        cls,
        meta: ManifestMeta,
        size: int,
        rng: Optional[np.random.Generator] = None,
        k_choices: Sequence[int] = (1, 2, 3),
        min_separation_deg: float = 3.0,
        array_errors: str = "mild",   # "none" | "mild" | "harsh"
    ) -> "SceneManifest":
        """Factory that fills a manifest with uniformly-random, valid scenes.

        Parameters
        ----------
        k_choices
            The set of ``n_sources`` values to sample uniformly.  For the
            thesis's SubspaceNet comparison we used ``(3,)``.
        min_separation_deg
            Minimum angular separation between any two sources.  Rejects
            conflicting draws up to 32 times before raising.
        array_errors
            Preset strength of ULA calibration errors:

            =========  ============================================
            preset     (gain, phase, coupling, position)
            =========  ============================================
            none       (0, 0, 0, 0)
            mild       (0.1 dB, 1°, 0.02, 1 %)
            harsh      (0.5 dB, 5°, 0.1, 5 %)
            =========  ============================================
        """
        rng = rng or np.random.default_rng()
        manifest = cls.create(meta, size)
        lo, hi = meta.angle_range_deg
        snr_lo, snr_hi = meta.snr_range_db
        err_presets = {
            "none":  (0.0, 0.0, 0.0, 0.0),
            "mild":  (0.1, 1.0, 0.02, 1.0),
            "harsh": (0.5, 5.0, 0.10, 5.0),
        }
        gain_err, phase_err, coupling, pos_err = err_presets[array_errors]

        for i in range(size):
            k = int(rng.choice(k_choices))
            for _attempt in range(32):
                angles = np.sort(rng.uniform(lo, hi, size=k).astype(np.float32))
                if k == 1 or np.min(np.diff(angles)) >= min_separation_deg:
                    break
            else:  # pragma: no cover
                raise RuntimeError(
                    f"Could not draw well-separated angles after 32 attempts "
                    f"(k={k}, min_sep={min_separation_deg}°)"
                )
            snr = float(rng.uniform(snr_lo, snr_hi))
            scene = Scene(
                seed=int(rng.integers(0, 2**32 - 1, dtype=np.uint64)),
                n_sources=k,
                angles_deg=np.pad(angles, (0, K_MAX - k), constant_values=ANGLE_FILL),
                snr_db=snr,
                array_type=ArrayType.ULA,
                modulation=ModulationType.NARROWBAND,
                gain_err_dB=gain_err,
                phase_err_deg=phase_err,
                mutual_coupling=coupling,
                position_err_pct=pos_err,
                scene_id=int(i),
            )
            manifest[i] = scene
        return manifest


__all__ = [
    "ArrayType",
    "ManifestMeta",
    "ModulationType",
    "Scene",
    "SceneManifest",
    "K_MAX",
]
