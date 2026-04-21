#!/usr/bin/env python3
"""Thin wrapper: ``python scripts/evaluate.py --config configs/eval/…``"""

from __future__ import annotations

import sys

from reconunet.cli.evaluate import main

if __name__ == "__main__":
    sys.exit(main())
