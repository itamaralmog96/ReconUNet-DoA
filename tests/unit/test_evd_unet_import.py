#!/usr/bin/env python3
"""
Test script to verify EVDUNet module imports work correctly.
"""

import sys
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

def test_direct_import():
    """Test importing directly from EVDUNet module"""
    print("🔧 Testing direct import from EVDUNet...")
    
    try:
        from models.deep_learning.EVDUNet import (
            CovarianceReconstructionUNet, 
            EVDCovarianceReconstructionUNet, 
            gram_diagonal_overload
        )
        print("  ✅ Direct import successful")
        return True
    except Exception as e:
        print(f"  ❌ Direct import failed: {e}")
        return False

def test_package_import():
    """Test importing through the package __init__"""
    print("🔧 Testing package import...")
    
    try:
        from models.deep_learning import (
            CovarianceReconstructionUNet, 
            EVDCovarianceReconstructionUNet, 
            gram_diagonal_overload
        )
        print("  ✅ Package import successful")
        return True
    except Exception as e:
        print(f"  ❌ Package import failed: {e}")
        return False

def test_model_creation():
    """Test creating model instances"""
    print("🔧 Testing model creation...")
    
    try:
        import torch
        from models.deep_learning.EVDUNet import (
            CovarianceReconstructionUNet, 
            EVDCovarianceReconstructionUNet
        )
        
        tau, M = 8, 8
        
        # Test CovarianceReconstructionUNet
        model1 = CovarianceReconstructionUNet(tau=tau, M=M)
        print(f"  ✅ CovarianceReconstructionUNet created: {sum(p.numel() for p in model1.parameters())} parameters")
        
        # Test EVDCovarianceReconstructionUNet
        model2 = EVDCovarianceReconstructionUNet(tau=tau, M=M)
        print(f"  ✅ EVDCovarianceReconstructionUNet created: {sum(p.numel() for p in model2.parameters())} parameters")
        
        return True
    except Exception as e:
        print(f"  ❌ Model creation failed: {e}")
        return False

def test_forward_pass():
    """Test forward pass through models"""
    print("🔧 Testing forward pass...")
    
    try:
        import torch
        from models.deep_learning.EVDUNet import (
            CovarianceReconstructionUNet, 
            EVDCovarianceReconstructionUNet
        )
        
        tau, M = 8, 8
        batch_size = 2
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, tau, 2*M, M)
        
        # Test CovarianceReconstructionUNet forward pass
        model1 = CovarianceReconstructionUNet(tau=tau, M=M)
        model1.eval()
        
        with torch.no_grad():
            outputs1 = model1(dummy_input)
            print(f"  ✅ CovarianceReconstructionUNet forward pass: {len(outputs1)} outputs")
            print(f"    - Output shapes: {[out.shape for out in outputs1]}")
        
        # Test EVDCovarianceReconstructionUNet forward pass
        model2 = EVDCovarianceReconstructionUNet(tau=tau, M=M)
        model2.eval()
        
        with torch.no_grad():
            outputs2 = model2(dummy_input)
            print(f"  ✅ EVDCovarianceReconstructionUNet forward pass: {len(outputs2)} outputs")
            print(f"    - Output shapes: {[out.shape for out in outputs2]}")
        
        return True
    except Exception as e:
        print(f"  ❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gram_diagonal_overload():
    """Test the helper function"""
    print("🔧 Testing gram_diagonal_overload function...")
    
    try:
        import torch
        from models.deep_learning.EVDUNet import gram_diagonal_overload
        
        # Create dummy complex matrix
        batch_size, M = 2, 8
        real_part = torch.randn(batch_size, M, M)
        imag_part = torch.randn(batch_size, M, M)
        K = torch.complex(real_part, imag_part)
        
        # Apply gram diagonal overload
        K_regularized = gram_diagonal_overload(K, eps=1.0)
        
        print(f"  ✅ gram_diagonal_overload successful: {K_regularized.shape}")
        print(f"    - Input shape: {K.shape}, Output shape: {K_regularized.shape}")
        
        # Check if result is Hermitian
        K_hermitian_check = torch.allclose(K_regularized, K_regularized.conj().transpose(-2, -1), atol=1e-6)
        print(f"    - Is Hermitian: {K_hermitian_check}")
        
        return True
    except Exception as e:
        print(f"  ❌ gram_diagonal_overload failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing EVDUNet Module Import and Functionality")
    print("=" * 60)
    
    tests = [
        test_direct_import,
        test_package_import,
        test_model_creation,
        test_forward_pass,
        test_gram_diagonal_overload
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()  # Add spacing between tests
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
            print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! EVDUNet module is ready for use.")
        print("\n🎯 You can now import the models using:")
        print("   from models.deep_learning.EVDUNet import CovarianceReconstructionUNet, EVDCovarianceReconstructionUNet")
        print("   or")
        print("   from models.deep_learning import CovarianceReconstructionUNet, EVDCovarianceReconstructionUNet")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())



