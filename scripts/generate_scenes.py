#!/usr/bin/env python3
"""Thin wrapper: ``python scripts/generate_scenes.py …``

Identical to the ``reconunet-generate`` console entry point, kept for users
who prefer the explicit ``scripts/`` invocation without ``pip install -e .``
first (e.g. CI runners using ``PYTHONPATH=src``).
"""

from __future__ import annotations

import sys

from reconunet.cli.generate_scenes import main

if __name__ == "__main__":
    sys.exit(main())
