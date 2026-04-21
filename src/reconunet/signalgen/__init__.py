"""
Signal generation and array processing module for DOA estimation.

This package provides:
- SignalGenerator: Generate realistic multi-source signals with various impairments
- ArrayConfig: Configuration for antenna array parameters
- ArrayModel: Antenna array model with hardware imperfections and steering vectors
- ReceivedSignal: Container for signals received at antenna arrays with noise handling
"""

from .signal_generator import SignalConfig, SignalGenerator
from .array_processing import ArrayConfig, ArrayModel, ReceivedSignal

__all__ = [
    'SignalConfig',
    'SignalGenerator', 
    'ArrayConfig',
    'ArrayModel',
    'ReceivedSignal'
]

__version__ = '1.0.0' 