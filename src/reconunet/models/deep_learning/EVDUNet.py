"""
EVD-Based UNet Architecture for DOA Estimation

This module contains UNet architectures for covariance matrix reconstruction and denoising,
including both standard covariance reconstruction and eigenvalue decomposition-based approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['gram_diagonal_overload', 'CovarianceReconstructionUNet', 'EVDCovarianceReconstructionUNet']

# Define helper functions for UNet
def gram_diagonal_overload(K, eps=1.0, batch_size=None):
    """
    Apply gram diagonal overloading to ensure positive semi-definite covariance matrix.
    
    Args:
        K: Complex tensor [batch_size, M, M]
        eps: Small positive value to add to diagonal
        batch_size: Batch size (optional)
    
    Returns:
        Positive semi-definite covariance matrix
    """
    if batch_size is None:
        batch_size = K.shape[0]
    
    M = K.shape[-1]
    
    # Ensure Hermitian symmetry
    K_hermitian = (K + torch.conj(K.transpose(-2, -1))) / 2
    
    # Add small positive value to diagonal for numerical stability
    eye = torch.eye(M, dtype=K.dtype, device=K.device).unsqueeze(0).expand(batch_size, -1, -1)
    K_regularized = K_hermitian + eps * eye
    
    return K_regularized

# CovarianceReconstructionUNet - Adapted from SubViT2
class CovarianceReconstructionUNet(nn.Module):
    """
    UNet for covariance matrix reconstruction and denoising.
    
    Takes autocorrelation tensors [batch, tau, 2M, M] and outputs clean covariance matrices.
    """
    
    def __init__(self, tau: int, M: int, activation_type: str = "relu", use_dropout: bool = True):
        """
        Initialize the Covariance Reconstruction UNet.
        
        Args:
            tau (int): Number of autocorrelation lags (input channels)
            M (int): Number of array elements
            activation_type (str): "relu" or "anti_rectifier" 
            use_dropout (bool): Whether to use dropout
        """
        super(CovarianceReconstructionUNet, self).__init__()
        
        self.tau = tau
        self.M = M
        self.activation_type = activation_type
        self.use_dropout = use_dropout
        
        # Define activation function and channel multiplier
        if activation_type == "relu":
            self.activation = nn.ReLU()
            self.anti_rect = False
            self.ch_mult = 1  # No channel doubling
        elif activation_type == "anti_rectifier":
            self.activation = nn.ReLU()  # Will be used in anti_rectifier method
            self.anti_rect = True
            self.ch_mult = 2  # Channel doubling from anti-rectifier
        else:
            raise ValueError(f"Unsupported activation_type: {activation_type}")
        
        # Dropout layer
        self.dropout = nn.Dropout(0.2) if use_dropout else nn.Identity()
        
        # Define base channel numbers - Match saved model architecture  
        ch1, ch2, ch3, ch4, ch5 = 16, 32, 64, 128, 256
        
        # Encoder layers - Double conv blocks with batch norm
        self.enc_conv1 = nn.Conv2d(tau, ch1, kernel_size=3, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(ch1 * self.ch_mult)
        self.enc_conv1_2 = nn.Conv2d(ch1 * self.ch_mult, ch1, kernel_size=3, padding=1)
        self.enc_bn1_2 = nn.BatchNorm2d(ch1 * self.ch_mult)
        
        self.enc_conv2 = nn.Conv2d(ch1 * self.ch_mult, ch2, kernel_size=3, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(ch2 * self.ch_mult)
        self.enc_conv2_2 = nn.Conv2d(ch2 * self.ch_mult, ch2, kernel_size=3, padding=1)
        self.enc_bn2_2 = nn.BatchNorm2d(ch2 * self.ch_mult)
        
        self.enc_conv3 = nn.Conv2d(ch2 * self.ch_mult, ch3, kernel_size=3, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(ch3 * self.ch_mult)
        self.enc_conv3_2 = nn.Conv2d(ch3 * self.ch_mult, ch3, kernel_size=3, padding=1)
        self.enc_bn3_2 = nn.BatchNorm2d(ch3 * self.ch_mult)
        
        # Pooling layers
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck layers
        self.bottleneck1 = nn.Conv2d(ch1 * self.ch_mult, ch1, kernel_size=3, padding=1)
        self.bn_bottleneck1 = nn.BatchNorm2d(ch1 * self.ch_mult)
        
        self.bottleneck2 = nn.Conv2d(ch2 * self.ch_mult, ch2, kernel_size=3, padding=1)
        self.bn_bottleneck2 = nn.BatchNorm2d(ch2 * self.ch_mult)
        
        self.bottleneck3 = nn.Conv2d(ch3 * self.ch_mult, ch3, kernel_size=3, padding=1)
        self.bn_bottleneck3 = nn.BatchNorm2d(ch3 * self.ch_mult)
        
        # Decoder layers
        self.upconv3 = nn.ConvTranspose2d(ch3 * self.ch_mult, ch2, kernel_size=2, stride=2)
        self.dec_conv3 = nn.Conv2d(ch2 + ch2 * self.ch_mult, ch2, kernel_size=3, padding=1)
        self.dec_bn3 = nn.BatchNorm2d(ch2 * self.ch_mult)
        self.dec_conv3_2 = nn.Conv2d(ch2 * self.ch_mult, ch2, kernel_size=3, padding=1)
        self.dec_bn3_2 = nn.BatchNorm2d(ch2 * self.ch_mult)
        
        self.upconv4 = nn.ConvTranspose2d(ch2 * self.ch_mult, ch1, kernel_size=2, stride=2)
        self.dec_conv4 = nn.Conv2d(ch1 + ch1 * self.ch_mult, ch1, kernel_size=3, padding=1)
        self.dec_bn4 = nn.BatchNorm2d(ch1 * self.ch_mult)
        self.dec_conv4_2 = nn.Conv2d(ch1 * self.ch_mult, ch1, kernel_size=3, padding=1)
        self.dec_bn4_2 = nn.BatchNorm2d(ch1 * self.ch_mult)
        
        # Final output layer
        self.final_conv = nn.Conv2d(ch1 * self.ch_mult, tau, kernel_size=1)
        
        # Transform from tau lags to single matrix
        self.tau_to_single_conv = nn.Conv2d(tau, 1, kernel_size=1, bias=True)
    
    def anti_rectifier(self, x):
        """Apply anti-rectifier activation: concat(ReLU(x), ReLU(-x))"""
        return torch.cat((self.activation(x), self.activation(-x)), dim=1)
    
    def apply_activation(self, x):
        """Apply the configured activation function."""
        if self.anti_rect:
            return self.anti_rectifier(x)
        else:
            return self.activation(x)
    
    def forward(self, Rx_tau: torch.Tensor):
        """
        Forward pass of the UNet.
        
        Args:
            Rx_tau (torch.Tensor): Input autocorrelation tensor [batch_size, tau, 2M, M]
            
        Returns:
            tuple: (Kx_raw, Rz, Rz_real_imag) - Raw output, complex covariance, real/imag format
        """
        batch_size = Rx_tau.shape[0]
        
        # Encoder block 1
        x1 = self.enc_conv1(Rx_tau)
        x1 = self.apply_activation(x1)
        x1 = self.enc_bn1(x1)
        x1 = self.enc_conv1_2(x1)
        x1 = self.apply_activation(x1)
        x1 = self.enc_bn1_2(x1)
        x1_skip = x1
        
        # Check dimensions for pooling
        h1, w1 = x1.shape[2], x1.shape[3]
        if h1 < 2 or w1 < 2:
            # Shallow network
            bottleneck = self.bottleneck1(x1)
            bottleneck_activated = self.apply_activation(bottleneck)
            bottleneck_activated = self.bn_bottleneck1(bottleneck_activated)
            dec4 = self.dec_conv4_2(bottleneck_activated)
            dec4 = self.apply_activation(dec4)
            dec4 = self.dec_bn4_2(dec4)
        else:
            # Encoder block 2
            x2 = self.pool1(x1)
            x2 = self.enc_conv2(x2)
            x2 = self.apply_activation(x2)
            x2 = self.enc_bn2(x2)
            x2 = self.enc_conv2_2(x2)
            x2 = self.apply_activation(x2)
            x2 = self.enc_bn2_2(x2)
            x2_skip = x2
            
            h2, w2 = x2.shape[2], x2.shape[3]
            if h2 < 2 or w2 < 2:
                # Use level 2 as bottleneck
                bottleneck = self.bottleneck2(x2)
                bottleneck_activated = self.apply_activation(bottleneck)
                bottleneck_activated = self.bn_bottleneck2(bottleneck_activated)
                
                up4 = self.upconv4(bottleneck_activated)
                concat4 = torch.cat([up4, x1_skip], dim=1)
                dec4 = self.dec_conv4(concat4)
                dec4 = self.apply_activation(dec4)
                dec4 = self.dec_bn4(dec4)
                dec4 = self.dec_conv4_2(dec4)
                dec4 = self.apply_activation(dec4)
                dec4 = self.dec_bn4_2(dec4)
            else:
                # Encoder block 3
                x3 = self.pool2(x2)
                x3 = self.enc_conv3(x3)
                x3 = self.apply_activation(x3)
                x3 = self.enc_bn3(x3)
                x3 = self.enc_conv3_2(x3)
                x3 = self.apply_activation(x3)
                x3 = self.enc_bn3_2(x3)
                
                # Bottleneck
                bottleneck = self.bottleneck3(x3)
                bottleneck_activated = self.apply_activation(bottleneck)
                bottleneck_activated = self.bn_bottleneck3(bottleneck_activated)
                
                # Decoder block 3
                up3 = self.upconv3(bottleneck_activated)
                concat3 = torch.cat([up3, x2_skip], dim=1)
                dec3 = self.dec_conv3(concat3)
                dec3 = self.apply_activation(dec3)
                dec3 = self.dec_bn3(dec3)
                dec3 = self.dec_conv3_2(dec3)
                dec3 = self.apply_activation(dec3)
                dec3 = self.dec_bn3_2(dec3)
                
                # Decoder block 4
                up4 = self.upconv4(dec3)
                concat4 = torch.cat([up4, x1_skip], dim=1)
                dec4 = self.dec_conv4(concat4)
                dec4 = self.apply_activation(dec4)
                dec4 = self.dec_bn4(dec4)
                dec4 = self.dec_conv4_2(dec4)
                dec4 = self.apply_activation(dec4)
                dec4 = self.dec_bn4_2(dec4)
        
        # Final output layers
        final = self.final_conv(dec4)
        final = self.dropout(final)
        
        # Transform from [B, tau, 2M, M] to [B, 1, 2M, M] then squeeze to [B, 2M, M]
        Kx_raw = self.tau_to_single_conv(final).squeeze(1)  # [B, 2M, M]
        
        # Convert to complex format
        M = self.M
        real_part = Kx_raw[:, :M, :]    # [B, M, M]
        imag_part = Kx_raw[:, M:, :]    # [B, M, M]
        Kx_complex = torch.complex(real_part, imag_part)  # [B, M, M]
        
        # Apply gram diagonal overload
        Rz = gram_diagonal_overload(Kx_complex, eps=1.0, batch_size=batch_size)
        
        # Convert back to real/imag format
        Rz_real_imag = torch.cat([torch.real(Rz), torch.imag(Rz)], dim=1)  # [B, 2M, M]
        
        return Kx_raw, Rz, Rz_real_imag

class CovarianceOnlyReconstructionUNet(nn.Module):
    """Ablation 08 — the covariance U-Net *without* eigenvalue / eigenvector heads.

    Same backbone as :class:`CovarianceReconstructionUNet`; the Hermitian,
    diagonally-loaded reconstructed covariance is the only learned output.

    * Training (``self.training``): returns ``{"K_recon": R_hat}`` so the
      trainer's composite loss reduces to L_rec (the ablation config lists only
      ``reconstruction_weight``).
    * Evaluation: returns the same ``(eigvals, eigvecs, R_hat)`` triple as
      :class:`EVDCovarianceReconstructionUNet`, with the eigenpairs obtained by
      ``eigh(R_hat)`` in descending order, so every downstream consumer (the
      Root-MUSIC back end, adapters, ablation metrics) works unchanged.  This is
      exactly the "covariance route" of the route comparison.
    """

    def __init__(self, tau: int, M: int, activation_type: str = "anti_rectifier",
                 use_dropout: bool = True):
        super().__init__()
        self.tau = tau
        self.M = M
        self.activation_type = activation_type
        self.use_dropout = use_dropout
        self.base_unet = CovarianceReconstructionUNet(tau, M, activation_type, use_dropout)

    def forward(self, Rx_tau: torch.Tensor):
        _, Rz, _ = self.base_unet(Rx_tau)                      # [B, M, M] complex, Hermitian + I
        if self.training:
            return {"K_recon": Rz}
        # fp64 CPU eigh: robust for these small clustered spectra (see
        # classical_batched.music); results go back to the model's device/dtype.
        w, V = torch.linalg.eigh(Rz.detach().to("cpu", torch.complex128))
        w = torch.flip(w, dims=[-1]).to(Rz.device, torch.float32)
        V = torch.flip(V, dims=[-1]).to(Rz.device, Rz.dtype)
        return w, V, Rz


# EVD-Based UNet Architecture
class EVDCovarianceReconstructionUNet(nn.Module):
    """
    Enhanced UNet that directly predicts eigenvalues and eigenvectors 
    instead of reconstructing the full covariance matrix.
    """
    
    def __init__(self, tau: int, M: int, activation_type: str = "anti_rectifier", use_dropout: bool = True):
        """
        Initialize the EVD-based Covariance Reconstruction UNet.
        
        Args:
            tau (int): Number of autocorrelation lags (input channels)
            M (int): Number of array elements
            activation_type (str): "relu" or "anti_rectifier" 
            use_dropout (bool): Whether to use dropout
        """
        super(EVDCovarianceReconstructionUNet, self).__init__()
        
        self.tau = tau
        self.M = M
        self.activation_type = activation_type
        self.use_dropout = use_dropout
        
        # Initialize base UNet (reuse existing architecture)
        self.base_unet = CovarianceReconstructionUNet(tau, M, activation_type, use_dropout)
        
        # EVD-specific components
        self.tau_compression = nn.Conv2d(tau, 1, kernel_size=1, bias=True)
        
        # Eigenvalue prediction head
        self.eigenvalue_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global pooling: [B, 1, 2M, M] → [B, 1, 1, 1]
            nn.Flatten(),                   # [B, 1]
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Dropout(0.1) if use_dropout else nn.Identity(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1) if use_dropout else nn.Identity(),
            nn.Linear(128, M),
            nn.ReLU()  # Ensure positive eigenvalues
        )
        
        # Eigenvector prediction head
        self.eigenvector_head = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1) if use_dropout else nn.Identity(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),  # Back to [B, 1, 2M, M]
            nn.Tanh()  # Bounded output for stability
        )
        
    def convert_to_complex_and_orthogonalize(self, eigenvecs_raw):
        """
        Convert [B, 2M, M] → [B, M, M] complex and enforce orthogonality.
        
        Args:
            eigenvecs_raw: [B, 2M, M] real/imaginary stacked format
            
        Returns:
            orthogonal_eigenvecs: [B, M, M] complex orthogonal matrix
        """
        B, two_M, M = eigenvecs_raw.shape
        
        # Split real and imaginary parts
        real_part = eigenvecs_raw[:, :M, :]     # [B, M, M]
        imag_part = eigenvecs_raw[:, M:, :]     # [B, M, M]
        
        # Create complex matrix
        eigenvecs_complex = torch.complex(real_part, imag_part)  # [B, M, M]
        
        # Enforce orthogonality with QR decomposition
        try:
            Q, R = torch.linalg.qr(eigenvecs_complex)
            
            # Handle sign consistency (make diagonal of R positive)
            R_diag = torch.diagonal(R, dim1=-2, dim2=-1)  # [B, M]
            signs = torch.sign(R_diag.real)  # Use real part for sign
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)  # Handle zeros
            Q_consistent = Q * signs.unsqueeze(-2)  # [B, M, M]
            # Extract the first row (first element of each eigenvector)
            first_elements = Q_consistent[:, 0, :]  # [B, M] - first element of each column
            
            # Compute the phase of the first elements
            first_phases = torch.angle(first_elements)  # [B, M]
            
            # Create phase correction factors: e^(-j * first_phase)
            phase_corrections = torch.exp(-1j * first_phases)  # [B, M]
            
            # Apply phase correction to each eigenvector (column)
            # Multiply each column by its corresponding phase correction
            Q_phase_referenced = Q_consistent * phase_corrections.unsqueeze(1)  # [B, M, M]
            
            return Q_phase_referenced
            
            
            
        except Exception as e:
            print(f"⚠️ QR decomposition failed: {e}, using Gram-Schmidt fallback")
            return self.gram_schmidt_orthogonalize(eigenvecs_complex)
    
    def gram_schmidt_orthogonalize(self, vectors):
        """
        Fallback Gram-Schmidt orthogonalization for batch of matrices.
        
        Args:
            vectors: [B, M, M] complex matrices
            
        Returns:
            orthogonal: [B, M, M] complex orthogonal matrices
        """
        B, M, _ = vectors.shape
        orthogonal = torch.zeros_like(vectors)
        
        for i in range(M):
            vi = vectors[:, :, i].clone()  # [B, M]
            
            # Subtract projections onto previous vectors
            for j in range(i):
                uj = orthogonal[:, :, j]  # [B, M]
                # Projection: <vi, uj> / <uj, uj> * uj
                inner_product = torch.sum(torch.conj(uj) * vi, dim=-1, keepdim=True)  # [B, 1]
                norm_squared = torch.sum(torch.conj(uj) * uj, dim=-1, keepdim=True).real  # [B, 1]
                projection = (inner_product / (norm_squared + 1e-8)) * uj  # [B, M]
                vi = vi - projection
            
            # Normalize
            norm = torch.norm(vi, dim=-1, keepdim=True)  # [B, 1]
            orthogonal[:, :, i] = vi / (norm + 1e-8)
        
        return orthogonal
    
    def forward(self, Rx_tau: torch.Tensor):
        """
        Forward pass of the EVD UNet.
        
        Args:
            Rx_tau (torch.Tensor): Input autocorrelation tensor [batch_size, tau, 2M, M]
            
        Returns:
            tuple: (eigenvals, eigenvecs, reconstructed_cov)
                - eigenvals: [B, M] real eigenvalues
                - eigenvecs: [B, M, M] complex orthogonal eigenvectors  
                - reconstructed_cov: [B, M, M] complex covariance matrix
        """
        batch_size = Rx_tau.shape[0]
        _, _, unet_features = self.base_unet(Rx_tau)  # Get UNet output
        # Compress tau dimension
        # compressed = self.tau_compression(unet_features)  # [B, 1, 2M, M]
        # Swap batch and channel dimensions if needed
        # Rx_tau: [batch_size, tau, 2M, M]  (should be [batch, channel, H, W] for UNet)
        # If batch and channel are swapped, permute them
        
        # Rx_tau is [batch_size, batch_size, 2M, M], which is wrong
        unet_features = unet_features.unsqueeze(1)
        # Predict eigenvalues
        eigenvals = self.eigenvalue_head(unet_features)  # [B, M]
        eigenvals = torch.sort(eigenvals, descending=True)[0]  # Sort in descending order
        
        # Predict eigenvectors
        eigenvecs_raw = self.eigenvector_head(unet_features)  # [B, 1, 2M, M]
        eigenvecs_raw = eigenvecs_raw.squeeze(1)  # [B, 2M, M]
        
        # Convert to complex and enforce orthogonality BEFORE sorting
        eigenvecs_orthogonal = self.convert_to_complex_and_orthogonalize(eigenvecs_raw)  # [B, M, M]
        
        # Sort eigenvalues in descending order and get indices
        eigenvals_sorted, sort_indices = torch.sort(eigenvals, descending=True, dim=-1)  # [B, M], [B, M]
        
        # Sort eigenvectors to match the sorted eigenvalues
        # Use gather to reorder eigenvector columns according to sort_indices
        batch_indices = torch.arange(batch_size, device=eigenvals.device).view(-1, 1, 1)  # [B, 1, 1]
        row_indices = torch.arange(self.M, device=eigenvals.device).view(1, -1, 1)      # [1, M, 1]
        col_indices = sort_indices.unsqueeze(1).expand(-1, self.M, -1)                     # [B, M, M]
        
        # Reorder eigenvector columns: eigenvecs_sorted[:, :, i] corresponds to eigenvals_sorted[:, i]
        eigenvecs_sorted = eigenvecs_orthogonal[batch_indices, row_indices, col_indices]     # [B, M, M]
        eigenvals_complex = eigenvals_sorted.to(dtype=torch.complex64)  # Convert to complex
        Lambda = torch.diag_embed(eigenvals_complex)  # [B, M, M] complex
        reconstructed_cov = (eigenvecs_sorted @ Lambda @ 
                           eigenvecs_sorted.conj().transpose(-2, -1))
        
        return eigenvals, eigenvecs_sorted, reconstructed_cov


if __name__ == "__main__":
    # Test the models when run directly
    print("✅ EVD-Based UNet Architecture Defined!")
    print("🏗️ Architecture Features:")
    print("   • Direct eigenvalue and eigenvector prediction")
    print("   • QR decomposition for orthogonality enforcement")
    print("   • Gram-Schmidt fallback for numerical stability")
    print("   • Automatic covariance reconstruction for compatibility")
    print("   • Supports both ReLU and anti-rectifier activations")
    
    # Quick test
    try:
        tau, M = 8, 8
        batch_size = 2
        
        # Test CovarianceReconstructionUNet
        model1 = CovarianceReconstructionUNet(tau=tau, M=M)
        dummy_input = torch.randn(batch_size, tau, 2*M, M)
        
        with torch.no_grad():
            outputs1 = model1(dummy_input)
            print(f"✅ CovarianceReconstructionUNet test passed: {len(outputs1)} outputs")
        
        # Test EVDCovarianceReconstructionUNet
        model2 = EVDCovarianceReconstructionUNet(tau=tau, M=M)
        
        with torch.no_grad():
            outputs2 = model2(dummy_input)
            print(f"✅ EVDCovarianceReconstructionUNet test passed: {len(outputs2)} outputs")
            
        print("🎯 All tests passed! Module is ready for import.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()