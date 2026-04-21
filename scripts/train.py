#!/usr/bin/env python3
"""Thin wrapper: ``python scripts/train.py --config configs/train/reconunet.yaml``.

Identical to the ``reconunet-train`` console entry point.
"""

from __future__ import annotations

import sys

from reconunet.cli.train import main

if __name__ == "__main__":
    sys.exit(main())
