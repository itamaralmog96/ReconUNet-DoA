"""Helper: build a Jupyter .ipynb file from a list of cells.

Each cell is a ("md" | "py", str).  String is stored as a list of lines
with trailing newlines, matching the Jupyter schema.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, Tuple


def _src(txt: str) -> list[str]:
    lines = txt.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        pass    # last line without newline is fine
    return lines


def build_notebook(cells: Iterable[Tuple[str, str]], out_path: Path) -> None:
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for kind, txt in cells:
        if kind == "md":
            nb["cells"].append({
                "cell_type": "markdown",
                "metadata": {},
                "source": _src(txt),
            })
        elif kind == "py":
            nb["cells"].append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": _src(txt),
            })
        else:
            raise ValueError(f"Unknown cell kind {kind!r}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"Wrote {out_path}  ({len(nb['cells'])} cells)")
