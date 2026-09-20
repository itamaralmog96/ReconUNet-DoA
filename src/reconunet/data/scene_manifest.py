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
    │ has_multipath    │ uint8  │   1  │ bool — enable multipath      │
    │ num_multipath    │ uint8  │   1  │ # of extra multipath paths   │
    │ mp_max_delay_fac │ float16│   2  │ legacy max_delay_factor      │
    │ mp_distribution  │ uint8  │   1  │ 0=uniform, 1=exponential     │
    │ _pad2            │ uint8  │  29  │ reserved for future fields   │
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
from tqdm.auto import tqdm

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
        # --- v1.1 multipath fields (zero-init = no multipath, back-compat) ---
        ("has_multipath", np.uint8),
        ("num_multipath", np.uint8),
        ("mp_max_delay_factor", np.float16),
        ("mp_distribution", np.uint8),
        ("_pad1", np.uint8, (29,)),
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
    # --- v1.1 multipath fields (default to "no multipath" for back-compat) ---
    has_multipath: bool = False
    num_multipath: int = 0
    mp_max_delay_factor: float = 10.0      # legacy default
    mp_distribution: int = 0               # 0 = uniform, 1 = exponential

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
    # Source bandwidth as a fraction of ``fs_Hz``.  The legacy
    # ``signalgen.SignalConfig`` default was 10 % of Nyquist = 0.05·fs, which
    # gives a coherence time of ~20 samples so that the few-sample multipath
    # delays leave the replica *highly correlated* with the direct path
    # (paper §II-B, |γ| ≈ 1).  ``0``/``None`` ⇒ white full-band sources (the
    # pre-v1.2 renderer), whose replicas were covariance-incoherent.
    source_bw_frac: Optional[float] = 0.05
    # Ablation knob (identifiability, reviewer R2-2): when set, EVERY scene is
    # rendered with the same array-error realisation drawn from this seed
    # (gain/phase/position/coupling jitter) instead of a per-scene draw.  The
    # per-scene rng stream is untouched, so sources / noise / multipath are
    # identical to the randomised rendering of the same scene.  ``None`` =
    # paper behaviour (randomised per scene).
    fixed_imperfection_seed: Optional[int] = None

    # --- sampling strategy ----------------------------------------------
    K_max: int = K_MAX
    angle_range_deg: Tuple[float, float] = (-60.0, 60.0)
    snr_range_db: Tuple[float, float] = (-10.0, 20.0)

    # --- provenance ------------------------------------------------------
    version: str = "1.0"
    created_utc: str = ""            # ISO 8601 timestamp, filled on save
    notes: str = ""

    def to_yaml(self, path: PathLike) -> None:
        array_type = self.array_type
        if hasattr(array_type, "name"):
            array_type = array_type.name
        payload: Dict[str, Any] = {
            "M": self.M,
            "element_spacing_lambda": self.element_spacing_lambda,
            "array_type": array_type,
            "T": self.T,
            "fs_Hz": self.fs_Hz,
            "speed_of_light": self.speed_of_light,
            "tau": self.tau,
            "source_bw_frac": self.source_bw_frac,
            "fixed_imperfection_seed": self.fixed_imperfection_seed,
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
        # Back-compat: old manifests have zero'd pad bytes, which deserialise
        # to has_multipath=False, num_multipath=0, mp_max_delay_factor=0.0.
        # If has_multipath is False we ignore the rest; if True and factor=0
        # we fall back to the legacy default of 10.0 at render time.
        has_mp = bool(int(r["has_multipath"]))
        mp_fac = float(r["mp_max_delay_factor"])
        if has_mp and mp_fac == 0.0:
            mp_fac = 10.0
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
            has_multipath=has_mp,
            num_multipath=int(r["num_multipath"]),
            mp_max_delay_factor=mp_fac,
            mp_distribution=int(r["mp_distribution"]),
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
        r["has_multipath"] = np.uint8(1 if scene.has_multipath else 0)
        r["num_multipath"] = np.uint8(max(0, int(scene.num_multipath)))
        r["mp_max_delay_factor"] = np.float16(float(scene.mp_max_delay_factor))
        r["mp_distribution"] = np.uint8(int(scene.mp_distribution))

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
        progress_desc: Optional[str] = None,
        # --- v1.1 multipath knobs (default off; legacy parity when enabled) ---
        enable_multipath: bool = False,
        max_paths: int = 3,                         # legacy default
        num_multipath_components: Optional[int] = None,
        multipath_distribution: str = "uniform",    # "uniform" | "exponential"
        mp_max_delay_factor: float = 10.0,          # legacy default
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
        _dist_map = {"uniform": 0, "exponential": 1}
        mp_dist_code = _dist_map[multipath_distribution]

        iterator = range(size)
        if progress_desc is not None:
            iterator = tqdm(iterator, desc=progress_desc, unit="scene", leave=True)
        for i in iterator:
            k = int(rng.choice(k_choices))
            for _attempt in range(200):
                angles = np.sort(rng.uniform(lo, hi, size=k).astype(np.float32))
                if k == 1 or np.min(np.diff(angles)) >= min_separation_deg:
                    break
            else:  # pragma: no cover
                raise RuntimeError(
                    f"Could not draw well-separated angles after 200 attempts "
                    f"(k={k}, min_sep={min_separation_deg}°)"
                )
            snr = float(rng.uniform(snr_lo, snr_hi))
            # Multipath component count — mirrors the legacy logic in
            # signal_generator.add_multipath: max_additional_paths = max_paths - k,
            # then either use the user-specified N or draw U[0, max_additional].
            if enable_multipath:
                max_additional = max(0, int(max_paths) - int(k))
                if num_multipath_components is None:
                    n_mp = int(rng.integers(0, max_additional + 1))
                else:
                    n_mp = min(int(num_multipath_components), max_additional)
                has_mp = bool(n_mp > 0)
            else:
                n_mp = 0
                has_mp = False
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
                has_multipath=has_mp,
                num_multipath=n_mp,
                mp_max_delay_factor=float(mp_max_delay_factor),
                mp_distribution=mp_dist_code,
            )
            manifest[i] = scene
        return manifest


    @classmethod
    def fixed_angles_snr_sweep(
        cls,
        meta: ManifestMeta,
        n_angle_configs: int,
        snr_levels_db: Sequence[float],
        rng: Optional[np.random.Generator] = None,
        k: int = 1,
        min_separation_deg: float = 10.0,
        array_errors: str = "mild",
        progress_desc: Optional[str] = None,
        # --- multipath knobs (default off) -----------------------------------
        enable_multipath: bool = False,
        num_multipath_components: int = 0,
        multipath_distribution: str = "uniform",
        mp_max_delay_factor: float = 10.0,
    ) -> "SceneManifest":
        """Paper §IV "Evaluation set construction (consistent angles across SNR)".

        Builds ``n_angle_configs * len(snr_levels_db)`` scenes laid out as
        ``n_angle_configs`` blocks of ``len(snr_levels_db)`` consecutive rows.
        Within one block the *scene seed*, the K direct-path angles, the
        per-element imperfection draws and the multipath layout are
        identical; only ``snr_db`` varies.  This matches the paper's
        construction:

            "1 000 unique angle configurations [...] are fixed and reused
             across all SNRs.  For each SNR, render one sample per
             configuration by adjusting only sigma_n^2."   (§IV, p.6)

        Each scenario is *narrow*: ``k`` and ``num_multipath_components``
        are scalars (not sets), so this call produces exactly the Basic /
        Moderate / Advanced-1 / Advanced-2 mixes from §IV.

        Parameters
        ----------
        n_angle_configs
            Number of unique direct-path angle configurations to draw
            (paper uses 1 000).
        snr_levels_db
            Sequence of SNRs (in dB) at which every angle configuration
            is rendered (paper uses [-20, -15, -10, -5, 0, 5, 10, 15, 20]).
        k
            Direct-source count per scene (Basic=1, Moderate=2, Crowded=4,
            OOD=1).
        num_multipath_components
            Coherent-replica count per scene (Basic=0, Moderate=1,
            Crowded=3, OOD=6).  ``enable_multipath`` is forced True iff
            this is > 0.
        """
        rng = rng or np.random.default_rng()
        S = len(snr_levels_db)
        if S == 0:
            raise ValueError("snr_levels_db must contain at least one entry")
        size = int(n_angle_configs) * S
        manifest = cls.create(meta, size)
        lo, hi = meta.angle_range_deg

        err_presets = {
            "none":  (0.0, 0.0, 0.0, 0.0),
            "mild":  (0.1, 1.0, 0.02, 1.0),
            "harsh": (0.5, 5.0, 0.10, 5.0),
        }
        gain_err, phase_err, coupling, pos_err = err_presets[array_errors]
        _dist_map = {"uniform": 0, "exponential": 1}
        mp_dist_code = _dist_map[multipath_distribution]

        n_mp = max(0, int(num_multipath_components))
        has_mp = bool(enable_multipath and n_mp > 0)

        iterator = range(int(n_angle_configs))
        if progress_desc is not None:
            iterator = tqdm(iterator, desc=progress_desc, unit="config",
                            leave=True)
        row = 0
        for cfg_idx in iterator:
            for _attempt in range(200):
                angles = np.sort(rng.uniform(lo, hi, size=k).astype(np.float32))
                if k == 1 or np.min(np.diff(angles)) >= min_separation_deg:
                    break
            else:  # pragma: no cover
                raise RuntimeError(
                    f"Could not draw well-separated angles after 200 attempts "
                    f"(k={k}, min_sep={min_separation_deg}°)"
                )
            # One seed per angle configuration: scene_seed is shared across
            # all S SNR levels for this configuration so source signals and
            # array imperfections are bit-identical and only the noise
            # variance changes (paper §IV, p.6).
            scene_seed = int(rng.integers(0, 2**32 - 1, dtype=np.uint64))
            angles_padded = np.pad(angles, (0, K_MAX - k), constant_values=ANGLE_FILL)
            for snr in snr_levels_db:
                scene = Scene(
                    seed=scene_seed,
                    n_sources=k,
                    angles_deg=angles_padded,
                    snr_db=float(snr),
                    array_type=ArrayType.ULA,
                    modulation=ModulationType.NARROWBAND,
                    gain_err_dB=gain_err,
                    phase_err_deg=phase_err,
                    mutual_coupling=coupling,
                    position_err_pct=pos_err,
                    scene_id=int(cfg_idx),       # block id = unique angle config
                    has_multipath=has_mp,
                    num_multipath=n_mp,
                    mp_max_delay_factor=float(mp_max_delay_factor),
                    mp_distribution=mp_dist_code,
                )
                manifest[row] = scene
                row += 1
        assert row == size
        return manifest


__all__ = [
    "ArrayType",
    "ManifestMeta",
    "ModulationType",
    "Scene",
    "SceneManifest",
    "K_MAX",
]
