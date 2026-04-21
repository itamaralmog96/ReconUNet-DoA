"""
Vision Transformer (ViT) for DOA estimation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from functools import partial
from collections import OrderedDict
from ..base_model import BaseModel
from ..model_registry import ModelRegistry

def drop_path(x: torch.Tensor, drop_prob: float = 0., training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    drop_prob : float
        Drop probability
    training : bool
        Whether in training mode
        
    Returns
    -------
    torch.Tensor
        Output tensor
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
    def __init__(self, drop_prob: float = None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_c: int = 3,
                 embed_dim: int = 768, norm_layer: Optional[nn.Module] = None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class Attention(nn.Module):
    """Multi-head self-attention module."""
    
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 qk_scale: Optional[float] = None, attn_drop_ratio: float = 0.,
                 proj_drop_ratio: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    """MLP as used in Vision Transformer."""
    
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, act_layer: nn.Module = nn.GELU,
                 drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    """Transformer block."""
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.,
                 qkv_bias: bool = False, qk_scale: Optional[float] = None,
                 drop_ratio: float = 0., attn_drop_ratio: float = 0.,
                 drop_path_ratio: float = 0., act_layer: nn.Module = nn.GELU,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, attn_drop_ratio=attn_drop_ratio,
                             proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

@ModelRegistry.register('vision_transformer')
class VisionTransformer(BaseModel):
    """Vision Transformer for DOA estimation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Vision Transformer model.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing:
            - num_sensors: int, number of sensors in the array
            - num_sources: int, number of sources to detect
            - start_angle: float, start angle of the search grid
            - end_angle: float, end angle of the search grid
            - step: float, step size of the search grid
            - img_size: int, input image size
            - patch_size: int, patch size
            - in_channels: int, number of input channels
            - embed_dim: int, embedding dimension
            - depth: int, depth of transformer
            - num_heads: int, number of attention heads
            - mlp_ratio: float, ratio of mlp hidden dim to embedding dim
            - qkv_bias: bool, enable bias for qkv
            - drop_ratio: float, dropout rate
            - attn_drop_ratio: float, attention dropout rate
            - drop_path_ratio: float, stochastic depth rate
        """
        super().__init__(config)
        
        self.num_sensors = config['num_sensors']
        self.num_sources = config['num_sources']
        
        # Create search grid
        self.register_buffer('_grid', torch.arange(
            config['start_angle'],
            config['end_angle'] + 0.0001,
            step=config['step']
        ))
        
        # Vision Transformer parameters
        self.embed_dim = config['embed_dim']
        self.depth = config['depth']
        self.num_heads = config['num_heads']
        self.mlp_ratio = config.get('mlp_ratio', 4.0)
        self.qkv_bias = config.get('qkv_bias', True)
        self.drop_ratio = config.get('drop_ratio', 0.)
        self.attn_drop_ratio = config.get('attn_drop_ratio', 0.)
        self.drop_path_ratio = config.get('drop_path_ratio', 0.)
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=config['img_size'],
            patch_size=config['patch_size'],
            in_c=config['in_channels'],
            embed_dim=self.embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=self.drop_ratio)
        
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_ratio, self.depth)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_ratio=self.drop_ratio,
                attn_drop_ratio=self.attn_drop_ratio,
                drop_path_ratio=dpr[i]
            )
            for i in range(self.depth)
        ])
        
        # Final layers
        self.norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, len(self._grid))
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vision Transformer.
        
        Parameters
        ----------
        x : torch.Tensor
            Input signal tensor of shape (batch_size, num_sensors, num_snapshots)
            
        Returns
        -------
        torch.Tensor
            Probability distribution over angles
        """
        # Reshape input to include real and imaginary parts
        x = torch.stack([x.real, x.imag], dim=1)
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        
        # Classification head
        x = self.head(x[:, 0])  # Use only class token
        
        return torch.sigmoid(x)
    
    def estimate_doa(self, x: torch.Tensor) -> np.ndarray:
        """
        Estimate DOA using the Vision Transformer.
        
        Parameters
        ----------
        x : torch.Tensor
            Input signal tensor of shape (batch_size, num_sensors, num_snapshots)
            
        Returns
        -------
        np.ndarray
            Estimated DOA angles
        """
        # Get probability distribution
        probs = self.forward(x)
        
        # Find peaks in the probability distribution
        doa_estimates = []
        for i in range(x.size(0)):
            peaks, _ = self._find_peaks(probs[i].cpu().numpy())
            doa_estimates.append(self._grid[peaks[:self.num_sources]].cpu().numpy())
            
        return np.array(doa_estimates)
    
    def _find_peaks(self, probs: np.ndarray, min_distance: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find peaks in the probability distribution.
        
        Parameters
        ----------
        probs : np.ndarray
            Probability distribution
        min_distance : int
            Minimum distance between peaks
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Peak indices and peak values
        """
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(probs, distance=min_distance)
        return peaks, properties['peak_heights']
    
    def _init_weights(self, m: nn.Module):
        """
        Initialize weights.
        
        Parameters
        ----------
        m : nn.Module
            Module to initialize
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    @property
    def grid(self) -> torch.Tensor:
        """
        Get the search grid.
        
        Returns
        -------
        torch.Tensor
            Search grid angles
        """
        return self._grid 