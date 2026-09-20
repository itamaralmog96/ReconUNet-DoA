#!/usr/bin/env python3
"""Full DA-MUSIC baseline training (one model per source count) — runnable
straight from the VSCode GUI.

HOW TO RUN
----------
1. Open this file in VSCode.
2. Make sure the project venv is the selected interpreter:
       Command Palette (Ctrl+Shift+P) -> "Python: Select Interpreter"
       -> pick  ./DOA_env/bin/python
3. Click the ▶ "Run Python File" button (top-right), or right-click in the
   editor -> "Run Python File in Terminal".

The published DA-MUSIC head has a fixed source count, so this trains FOUR
instances in sequence — configs/train/damusic_paper_k{1,2,3,4}.yaml — each on
the K-subset of the 2M paper corpus with the shared paper schedule.  Each run
writes to experiments/runs/damusic_paper/k<K>/ (checkpoints/best.pt, tb/,
history.json) and a timestamped log is kept per K.

  NOTE: this OVERWRITES experiments/runs/damusic_paper/k<K>/ for every K in
  K_LIST.  Back up first if you need a previous run.

Companion runners: run_subspacenet_full_training.py, run_subvit_full_training.py.
Do NOT co-run with SubspaceNet training (its eigh-based head is starved by any
other GPU job on this box); DA-MUSIC uses the same eigh path.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Knobs — edit these, then just hit Run.
# ---------------------------------------------------------------------------
# Cap the number of epochs per K. None = full paper recipe (up to 300 epochs,
# early-stopping patience 25). Set to a small int (e.g. 1) for a smoke run.
EPOCHS: int | None = None

# Which source counts to train (each is an independent model).
K_LIST: tuple[int, ...] = (1, 2, 3, 4)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = REPO_ROOT / "experiments/runs/damusic_paper"


class _Tee:
    """Mirror writes to a real stream and a log file."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, data):
        self._stream.write(data)
        self._fh.write(data)
        return len(data)

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _print_gpu_banner() -> None:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        print(f"[run] WARNING: could not import torch ({exc}); is the DOA_env "
              f"interpreter selected in VSCode?")
        return
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"[run] GPU: {name} ({total:.0f} GiB), torch {torch.__version__} "
              f"-> training will run on CUDA.")
    else:
        print(f"[run] WARNING: torch {torch.__version__} reports NO CUDA device "
              f"-> training will fall back to CPU (very slow).")


def _train_one(k: int) -> int:
    from reconunet.cli.train import main as train_main

    cfg = REPO_ROOT / f"configs/train/damusic_paper_k{k}.yaml"
    run_dir = RUN_ROOT / f"k{k}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = run_dir / f"train_{stamp}.log"
    log_fh = open(log_path, "w", buffering=1)
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_out, log_fh)
    sys.stderr = _Tee(orig_err, log_fh)
    try:
        print(f"[run] ===== DA-MUSIC K={k} =====")
        print(f"[run] config:  {cfg.relative_to(REPO_ROOT)}  "
              f"(paper_corpus K={k} subset, Adam 1e-4, batch 2048)")
        print(f"[run] outputs: {run_dir.relative_to(REPO_ROOT)}")
        print(f"[run] log:     {log_path.relative_to(REPO_ROOT)}")
        if EPOCHS is not None:
            print(f"[run] epochs override: {EPOCHS}")
        argv = ["--config", str(cfg)]
        if EPOCHS is not None:
            argv += ["--epochs", str(EPOCHS)]
        rc = train_main(argv)
        print(f"[run] K={k} finished with rc={rc}; best checkpoint: "
              f"{(run_dir / 'checkpoints/best.pt').relative_to(REPO_ROOT)}")
        return rc
    finally:
        sys.stdout, sys.stderr = orig_out, orig_err
        log_fh.close()


def main() -> int:
    os.chdir(REPO_ROOT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[run] repo: {REPO_ROOT}")
    print(f"[run] training DA-MUSIC for K in {K_LIST}")
    _print_gpu_banner()
    worst = 0
    for k in K_LIST:
        rc = _train_one(k)
        worst = max(worst, rc)
        if rc != 0:
            print(f"[run] K={k} exited non-zero ({rc}); continuing with the next K.")
    print(f"[run] all done. results under {RUN_ROOT}")
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
