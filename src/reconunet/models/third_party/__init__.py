"""Adapters around baseline DOA estimators that live as git submodules.

The submodules under ``third_party/`` are vendored from upstream repositories
*as-is*; we do not modify their source.  Each adapter bridges the upstream
API to the shared ReconUNet interface defined below, so all three models
(ReconUNet, SubspaceNet, SubViT) can be driven by one training config and
one evaluation harness.

Expected project layout when adapters are used:

    <repo-root>/ReconUNet/
    ├── src/reconunet/models/third_party/
    │   ├── _base.py           <-- BaselineAdapter abstract class
    │   ├── subspacenet_adapter.py
    │   └── subvit_adapter.py
    └── third_party/           <-- git submodules (DO NOT MODIFY)
        ├── subspacenet/       <-- https://github.com/ShlezingerLab/SubspaceNet.git
        └── doa_est_master/    <-- https://github.com/zzb-nice/DOA_est_Master.git

Adapters resolve the submodule paths by walking upward from this file; a
runtime ``sys.path`` insertion is necessary because the vendored repos are
not packages themselves.
"""

from ._base import BaselineAdapter, BaselineOutput

__all__ = ["BaselineAdapter", "BaselineOutput"]
