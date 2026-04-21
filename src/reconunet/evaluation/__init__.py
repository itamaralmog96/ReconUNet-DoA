# ReconUNet evaluation package

"""High-level evaluation utilities and sub-modules.

The :pymod:`reconunet.evaluation.evaluation` module provides a unified
interface that wraps the classic, deep-learning and hybrid evaluation
routines into a single class.

Sub-packages:
    classic             - Classic DoA algorithms (MUSIC, MVDR, Root-MUSIC, ESPRIT)
    monte_carlo         - Benchmark scripts with Monte-Carlo sweeps
    dataset_evaluation  - Per-dataset evaluation harnesses
"""

from importlib import import_module as _imp

# Re-export the unified evaluator for convenience -----------------------------
try:
    _evaluator_mod = _imp("reconunet.evaluation.evaluation")
    DOAEvaluator = _evaluator_mod.DOAEvaluator  # type: ignore[attr-defined]
    EvaluationConfig = _evaluator_mod.EvaluationConfig  # type: ignore[attr-defined]
except Exception:
    # Lazy import failed (cyclic during init) – ignore.
    pass
