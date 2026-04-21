"""Data subsystem: dataset generation, scene manifest, renderer, and dataloaders.

The modern entry-point for training-data production is
:class:`reconunet.data.scene_manifest.SceneManifest` plus
:class:`reconunet.data.scene_renderer.SceneRenderer`.  The older
``dataset_generator`` / ``controlled_dataset_generator`` modules remain for
backwards compatibility with pre-reorg notebooks and checkpoints.
"""
