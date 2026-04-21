"""Command-line entry points for the :mod:`reconunet` package.

These modules are wired up in ``pyproject.toml`` under ``[project.scripts]``,
so after ``pip install -e .`` you get three console commands:

.. code-block:: bash

    reconunet-generate --config configs/data/shared_manifest.yaml
    reconunet-train    --config configs/train/reconunet.yaml
    reconunet-evaluate --config configs/eval/evaluation_config.yaml

Each module exposes a ``main()`` that parses ``argv`` and returns an int
exit code, so they double as library functions.
"""

__all__ = ["generate_scenes", "train", "evaluate"]
