"""Training package for Tri4Net subspace models.

This package provides training functionality for various subspace-based
deep learning models, with both standard and performance-optimized implementations.
"""

# Make key classes available at package level
try:
    from .subspace_training import SubspaceTrainer, train_subspace_model
    from .fast_training import FastTrainer, train_fast_subspace_net, train_simple_cnn, main as fast_train_main
    
    __all__ = [
        'SubspaceTrainer',
        'train_subspace_model', 
        'FastTrainer',
        'train_fast_subspace_net',
        'train_simple_cnn',
        'fast_train_main'
    ]
except ImportError:
    # Graceful degradation if dependencies are missing
    __all__ = [] 