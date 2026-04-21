"""Scenario Classification Models for DoA Data.

This module implements various deep learning architectures for classifying
scenarios in DoA estimation datasets. The models classify received signals
into 36 different scenarios based on:
- 6 source/multipath combinations
- 2 array conditions (perfect vs. imperfect) 
- 3 SNR levels (low/mid/high)

Models implemented:
- ScenarioClassificationCNN: Multi-scale CNN (recommended)
- CovarianceResNet: ResNet processing covariance matrices
- SignalTransformer: Transformer with spatial-temporal attention
- HybridCNNRNN: CNN-RNN hybrid for efficiency
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

__all__ = [
    "ScenarioClassificationCNN", 
    "CovarianceResNet", 
    "SignalTransformer", 
    "HybridCNNRNN"
]


class ScenarioClassificationCNN(nn.Module):
    """Multi-scale CNN for scenario classification.
    
    Processes received signals with multi-scale spatial and temporal convolutions
    to classify into 36 different scenarios.
    
    Args:
        N: Number of array elements (default: 8)
        T: Number of time samples (default: 1024)
        num_classes: Number of scenario classes (default: 36)
        dropout_rate: Dropout rate for regularization (default: 0.5)
    """
    
    def __init__(self, N: int = 8, T: int = 1024, num_classes: int = 36, dropout_rate: float = 0.5):
        super().__init__()
        self.N = N
        self.T = T
        self.num_classes = num_classes
        
        # Multi-scale spatial processing (across antennas)
        self.spatial_conv = nn.Sequential(
            # First spatial scale
            nn.Conv2d(2, 64, kernel_size=(max(1, N//2), 1), padding=(max(0, N//4), 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Second spatial scale
            nn.Conv2d(64, 128, kernel_size=(max(1, N//4), 1), padding=(max(0, N//8), 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Multi-scale temporal processing 
        self.temporal_conv = nn.Sequential(
            # First temporal scale
            nn.Conv2d(128, 256, kernel_size=(1, 32), stride=(1, 4), padding=(0, 14)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Second temporal scale
            nn.Conv2d(256, 512, kernel_size=(1, 16), stride=(1, 4), padding=(0, 6)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Third temporal scale
            nn.Conv2d(512, 512, kernel_size=(1, 8), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Global pooling + classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Complex received signals [B, N, T]
            
        Returns:
            Logits for 36 scenario classes [B, 36]
        """
        B, N, T = x.shape
        
        # Convert complex to real representation [B, 2, N, T]
        x_real = torch.stack([x.real, x.imag], dim=1)
        
        # Multi-scale spatial processing
        features = self.spatial_conv(x_real)
        
        # Multi-scale temporal processing
        features = self.temporal_conv(features)
        
        # Classification
        logits = self.classifier(features)
        
        return logits


class CovarianceResNet(nn.Module):
    """ResNet processing covariance matrices for scenario classification.
    
    Computes covariance matrices from received signals and processes them
    with a ResNet backbone for scenario classification.
    
    Args:
        N: Number of array elements (default: 8)
        num_classes: Number of scenario classes (default: 36)
        pretrained: Whether to use pretrained ResNet weights (default: False)
        resnet_variant: ResNet variant ('resnet18', 'resnet34', 'resnet50')
    """
    
    def __init__(self, N: int = 8, num_classes: int = 36, 
                 pretrained: bool = False, resnet_variant: str = 'resnet18'):
        super().__init__()
        self.N = N
        self.num_classes = num_classes
        
        # Load ResNet backbone
        if resnet_variant == 'resnet18':
            self.backbone = torchvision.models.resnet18(pretrained=pretrained)
            fc_features = 512
        elif resnet_variant == 'resnet34':
            self.backbone = torchvision.models.resnet34(pretrained=pretrained)
            fc_features = 512
        elif resnet_variant == 'resnet50':
            self.backbone = torchvision.models.resnet50(pretrained=pretrained)
            fc_features = 2048
        else:
            raise ValueError(f"Unsupported ResNet variant: {resnet_variant}")
        
        # Modify first conv layer for 3-channel covariance input
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify final classifier
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(fc_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Covariance preprocessing layers
        self.cov_prep = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1)
        )
        
        # If N < 224, we need upsampling for ResNet
        if N < 224:
            self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        else:
            self.upsample = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Complex received signals [B, N, T]
            
        Returns:
            Logits for 36 scenario classes [B, 36]
        """
        B, N, T = x.shape
        
        # Compute sample covariance matrix
        Rx = torch.matmul(x, x.conj().transpose(-1, -2)) / T  # [B, N, N]
        
        # Convert to 3-channel representation [B, 3, N, N]
        features = torch.stack([
            Rx.real,           # Real part
            Rx.imag,           # Imaginary part
            torch.abs(Rx)      # Magnitude
        ], dim=1)
        
        # Preprocess covariance features
        features = self.cov_prep(features)
        
        # Upsample if needed for ResNet input size
        features = self.upsample(features)
        
        # ResNet classification
        logits = self.backbone(features)
        
        return logits


class SignalTransformer(nn.Module):
    """Transformer for scenario classification with spatial-temporal attention.
    
    Uses transformer architecture to capture long-range dependencies in
    received signals for scenario classification.
    
    Args:
        N: Number of array elements (default: 8)
        T: Number of time samples (default: 1024)
        num_classes: Number of scenario classes (default: 36)
        d_model: Model dimension (default: 256)
        nhead: Number of attention heads (default: 8)
        num_layers: Number of transformer layers (default: 6)
        patch_size: Temporal patch size (default: 64)
    """
    
    def __init__(self, N: int = 8, T: int = 1024, num_classes: int = 36,
                 d_model: int = 256, nhead: int = 8, num_layers: int = 6,
                 patch_size: int = 64):
        super().__init__()
        self.N = N
        self.T = T
        self.num_classes = num_classes
        self.d_model = d_model
        self.patch_size = patch_size
        self.num_patches = T // patch_size
        
        # Patch embedding (real + imag components)
        self.patch_embed = nn.Linear(patch_size * 2, d_model)
        
        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, N * self.num_patches, d_model))
        self.dropout = nn.Dropout(0.1)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Complex received signals [B, N, T]
            
        Returns:
            Logits for 36 scenario classes [B, 36]
        """
        B, N, T = x.shape
        
        # Convert to real representation and create patches
        x_real = torch.stack([x.real, x.imag], dim=-1)  # [B, N, T, 2]
        x_patches = x_real.reshape(B, N, self.num_patches, self.patch_size * 2)
        x_patches = x_patches.reshape(B, N * self.num_patches, self.patch_size * 2)
        
        # Embed patches
        tokens = self.patch_embed(x_patches)  # [B, N*num_patches, d_model]
        tokens = tokens + self.pos_embed
        tokens = self.dropout(tokens)
        
        # Transformer processing
        tokens = self.transformer(tokens)
        
        # Global average pooling across all tokens
        features = tokens.mean(dim=1)  # [B, d_model]
        
        # Classification
        logits = self.classifier(features)
        
        return logits


class HybridCNNRNN(nn.Module):
    """Hybrid CNN-RNN for efficient scenario classification.
    
    Combines CNN for spatial feature extraction with RNN for temporal modeling.
    Optimized for fast training and inference.
    
    Args:
        N: Number of array elements (default: 8)
        T: Number of time samples (default: 1024)
        num_classes: Number of scenario classes (default: 36)
        rnn_hidden_size: RNN hidden dimension (default: 128)
        rnn_num_layers: Number of RNN layers (default: 2)
        temporal_downsample: Temporal downsampling factor (default: 2)
    """
    
    def __init__(self, N: int = 8, T: int = 1024, num_classes: int = 36,
                 rnn_hidden_size: int = 128, rnn_num_layers: int = 2,
                 temporal_downsample: int = 2):
        super().__init__()
        self.N = N
        self.T = T
        self.num_classes = num_classes
        self.temporal_downsample = temporal_downsample
        
        # CNN for spatial feature extraction
        self.spatial_cnn = nn.Sequential(
            # Stack real/imag channels: 2*N input channels
            nn.Conv1d(2*N, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        
        # Temporal downsampling
        downsample_size = T // temporal_downsample
        self.temporal_pool = nn.AdaptiveAvgPool1d(downsample_size)
        
        # RNN for temporal modeling
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
            dropout=0.2 if rnn_num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden_size * 2, 256),  # *2 for bidirectional
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.normal_(param.data)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Complex received signals [B, N, T]
            
        Returns:
            Logits for 36 scenario classes [B, 36]
        """
        B, N, T = x.shape
        
        # Stack real and imaginary parts: [B, 2*N, T]
        x_real = torch.cat([x.real, x.imag], dim=1)
        
        # CNN spatial processing
        features = self.spatial_cnn(x_real)  # [B, 256, T]
        
        # Temporal downsampling
        features = self.temporal_pool(features)  # [B, 256, T//downsample]
        
        # Prepare for RNN: [B, T//downsample, 256]
        features = features.transpose(1, 2)
        
        # RNN temporal processing
        rnn_output, (h_n, c_n) = self.rnn(features)
        
        # Use final hidden state from both directions
        # h_n shape: [num_layers*2, B, hidden_size] for bidirectional
        final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [B, hidden_size*2]
        
        # Classification
        logits = self.classifier(final_hidden)
        
        return logits


# Optional registration with Tri4Net ModelRegistry
try:
    from ..model_registry import ModelRegistry  # type: ignore
    
    ModelRegistry.register("scenario_cnn")(ScenarioClassificationCNN)
    ModelRegistry.register("scenario_resnet")(CovarianceResNet)
    ModelRegistry.register("scenario_transformer")(SignalTransformer)
    ModelRegistry.register("scenario_hybrid")(HybridCNNRNN)
    
except ImportError:
    pass  # ModelRegistry is optional 