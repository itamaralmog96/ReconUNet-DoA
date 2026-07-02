#!/usr/bin/env python3
"""Full SubViT (DOA-ViT) training — runnable straight from the VSCode GUI.

HOW TO RUN
----------
1. Open this file in VSCode.
2. Make sure the project venv is the selected interpreter:
       Command Palette (Ctrl+Shift+P) -> "Python: Select Interpreter"
       -> pick  ./DOA_env/bin/python
3. Click the ▶ "Run Python File" button (top-right), or right-click in the
   editor -> "Run Python File in Terminal".

Companion to scripts/run_subspacenet_full_training.py — same in-process trainer
call, same GPU banner + tee-to-log, but for the SubViT baseline. It trains
SubViT on the 2M paper corpus (configs/train/subvit_paper.yaml).

There is no SubViT-specific per-K comparison script (the SubspaceNet one,
eval_subnet_genk.py, is hard-wired to SubspaceNet). The apples-to-apples 3-way
comparison (ReconUNet vs SubspaceNet vs SubViT) is a separate downstream step
once all three checkpoints exist — see the note printed at the end.

Output is shown live in the terminal AND saved to a timestamped log under
experiments/runs/subvit_paper/.

  NOTE: this OVERWRITES experiments/runs/subvit_paper/ (checkpoints, tb,
  history). Back it up first if you need to keep a previous run.

  GPU SHARING: if the SubspaceNet full run is still going, this will share the
  one GPU with it — both fit in 16 GiB but throughput drops for both. For the
  fastest run, start this after SubspaceNet finishes (or just for a capped
  smoke test, set EPOCHS below).
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Knobs — edit these, then just hit Run.
# ---------------------------------------------------------------------------
# Cap the number of epochs. None = full paper recipe (up to 300 epochs with
# early-stopping patience 25). Set to a small int (e.g. 2) for a quick smoke
# test of the whole pipeline on the real corpus.
EPOCHS: int | None = None
# ---------------------------------------------------------------------------

# scripts/ lives one level below the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CONFIG = REPO_ROOT / "configs/train/subvit_paper.yaml"
RUN_DIR = REPO_ROOT / "experiments/runs/subvit_paper"
CKPT = RUN_DIR / "checkpoints/best.pt"


class _Tee:
    """Mirror writes to a real stream and a log file (so the GUI terminal and
    the on-disk log both get everything, including tqdm \\r progress lines)."""

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

    def __getattr__(self, name):  # isatty(), fileno(), encoding, ...
        return getattr(self._stream, name)


def _print_gpu_banner() -> None:
    """Confirm up-front whether training will run on the GPU."""
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


def main() -> int:
    # Relative paths inside the YAML resolve from the repo root.
    os.chdir(REPO_ROOT)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = RUN_DIR / f"train_{stamp}.log"
    log_fh = open(log_path, "w", buffering=1)  # line-buffered

    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_out, log_fh)
    sys.stderr = _Tee(orig_err, log_fh)

    try:
        print(f"[run] repo:    {REPO_ROOT}")
        print(f"[run] config:  {TRAIN_CONFIG.relative_to(REPO_ROOT)}  "
              f"(2M paper_corpus, SubViT, Adam 1e-4, batch 512)")
        print(f"[run] outputs: {RUN_DIR.relative_to(REPO_ROOT)}")
        print(f"[run] log:     {log_path.relative_to(REPO_ROOT)}")
        if EPOCHS is not None:
            print(f"[run] epochs override: {EPOCHS}")
        _print_gpu_banner()

        # ---- training (in-process; this can take several hours) ------------
        print("[run] ===== training (this can take several hours) =====")
        from reconunet.cli.train import main as train_main

        argv = ["--config", str(TRAIN_CONFIG)]
        if EPOCHS is not None:
            argv += ["--epochs", str(EPOCHS)]
        rc = train_main(argv)
        if rc != 0:
            print(f"[run] training exited non-zero ({rc}).")
            return rc
        print(f"[run] ===== training done; best checkpoint: {CKPT} =====")

        print("[run] all done.")
        print("[run] For the 3-way comparison (ReconUNet vs SubspaceNet vs "
              "SubViT), once all three checkpoints exist run:")
        print("[run]     reconunet-evaluate --config configs/eval/full.yaml")
        return 0
    finally:
        sys.stdout, sys.stderr = orig_out, orig_err
        log_fh.close()


if __name__ == "__main__":
    raise SystemExit(main())
