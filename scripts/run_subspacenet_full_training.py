#!/usr/bin/env python3
"""Full SubspaceNet training — runnable straight from the VSCode GUI.

HOW TO RUN
----------
1. Open this file in VSCode.
2. Make sure the project venv is the selected interpreter:
       Command Palette (Ctrl+Shift+P) -> "Python: Select Interpreter"
       -> pick  ./DOA_env/bin/python
3. Click the ▶ "Run Python File" button (top-right), or right-click in the
   editor -> "Run Python File in Terminal".

This is the in-process equivalent of scripts/run_subspacenet_full_training.sh
(it calls the trainer directly instead of the `reconunet-train` CLI, so it
works from the GUI without any PATH setup). It:

  1. trains SubspaceNet on the 2M paper corpus (configs/train/subspacenet_paper.yaml),
  2. then runs the per-K head-to-head vs ReconUNet + classical baselines.

Output is shown live in the terminal AND saved to a timestamped log under
experiments/runs/subspacenet_paper/.

  NOTE: this OVERWRITES experiments/runs/subspacenet_paper/ (checkpoints, tb,
  history). Back it up first if you need to keep a previous run.
"""

from __future__ import annotations

import os
import subprocess
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

# Run the per-K SubspaceNet-vs-ReconUNet comparison after training finishes.
RUN_COMPARISON: bool = True
# ---------------------------------------------------------------------------

# scripts/ lives one level below the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CONFIG = REPO_ROOT / "configs/train/subspacenet_paper.yaml"
RUN_DIR = REPO_ROOT / "experiments/runs/subspacenet_paper"
CKPT = RUN_DIR / "checkpoints/best.pt"
RECONUNET_CKPT = REPO_ROOT / "experiments/runs/reconunet_paper/checkpoints/best.pt"
TEST_MANIFEST = REPO_ROOT / "data/scenes/subnet_genk/test.npy"


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
    # Relative paths inside the YAML / comparison script resolve from the root.
    os.chdir(REPO_ROOT)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = RUN_DIR / f"train_{stamp}.log"
    log_fh = open(log_path, "w", buffering=1)  # line-buffered

    # Tee stdout/stderr so everything printed below (and the trainer's tqdm
    # bars, which go to stderr) lands in both the terminal and the log file.
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_out, log_fh)
    sys.stderr = _Tee(orig_err, log_fh)

    try:
        print(f"[run] repo:    {REPO_ROOT}")
        print(f"[run] config:  {TRAIN_CONFIG.relative_to(REPO_ROOT)}  "
              f"(2M paper_corpus, variable-K, Adam 1e-4, batch 2048)")
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
            print(f"[run] training exited non-zero ({rc}); skipping comparison.")
            return rc
        print(f"[run] ===== training done; best checkpoint: {CKPT} =====")

        # ---- per-K head-to-head vs ReconUNet + classical baselines ---------
        if not RUN_COMPARISON:
            print("[run] comparison disabled (RUN_COMPARISON=False).")
        elif CKPT.is_file() and RECONUNET_CKPT.is_file() and TEST_MANIFEST.is_file():
            print("[run] ===== per-K comparison vs ReconUNet =====")
            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts/analysis/eval_subnet_genk.py"),
                "--checkpoint", str(CKPT),
                "--manifest", str(TEST_MANIFEST),
                "--reconunet-checkpoint", str(RECONUNET_CKPT),
                "--output-dir", str(RUN_DIR),
            ]
            # Stream the subprocess output through our Tee (line by line).
            proc = subprocess.Popen(
                cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
            ret = proc.wait()
            if ret != 0:
                print(f"[run] comparison exited non-zero ({ret}).")
        else:
            missing = [str(p) for p in (CKPT, RECONUNET_CKPT, TEST_MANIFEST)
                       if not p.is_file()]
            print(f"[run] skipping comparison (missing: {', '.join(missing)})")

        print(f"[run] all done. results in {RUN_DIR}")
        return 0
    finally:
        sys.stdout, sys.stderr = orig_out, orig_err
        log_fh.close()


if __name__ == "__main__":
    raise SystemExit(main())
