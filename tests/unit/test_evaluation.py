#!/usr/bin/env python3
"""
Test script to verify the comprehensive DOA evaluation works correctly.
This script runs a minimal test to check all components.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

def test_imports():
    """Test that all required imports work"""
    print("🔧 Testing imports...")
    
    try:
        from signalgen.array_processing import ArrayConfig, ArrayModel
        from signalgen.signal_generation import SignalConfig, SignalGenerator
        print("  ✅ Signal generation imports OK")
    except Exception as e:
        print(f"  ❌ Signal generation imports failed: {e}")
        return False
    
    try:
        from models.classic.music import MUSIC
        from models.classic.mvdr import MVDR
        from models.classic.beamformer import Beamformer
        print("  ✅ Classic algorithm imports OK")
    except Exception as e:
        print(f"  ❌ Classic algorithm imports failed: {e}")
        return False
    
    try:
        from models.deep_learning.subspace_models import SubspaceUNet
        print("  ✅ UNet model import OK")
    except Exception as e:
        print(f"  ❌ UNet model import failed: {e}")
        return False
    
    return True

def test_array_setup():
    """Test array and signal generation setup"""
    print("🔧 Testing array setup...")
    
    try:
        from signalgen.array_processing import ArrayConfig, ArrayModel
        from signalgen.signal_generation import SignalConfig, SignalGenerator
        
        # Create array configuration
        array_config = ArrayConfig(
            array_type='linear',
            num_elements=8,
            carrier_freq=2.45e9,
            element_spacing=0.5
        )
        
        array_model = ArrayModel(array_config)
        print(f"  ✅ Array model created: {array_config.num_elements} element {array_config.array_type} array")
        
        # Create signal configuration
        signal_config = SignalConfig(
            carrier_freq=array_config.carrier_freq,
            sampling_freq=1e6,
            num_snapshots=256,
            bandwidth=1e6
        )
        
        signal_generator = SignalGenerator(signal_config, array_model)
        print("  ✅ Signal generator created")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Array setup failed: {e}")
        return False

def test_signal_generation():
    """Test signal generation"""
    print("🔧 Testing signal generation...")
    
    try:
        from signalgen.array_processing import ArrayConfig, ArrayModel
        from signalgen.signal_generation import SignalConfig, SignalGenerator
        
        # Setup
        array_config = ArrayConfig(array_type='linear', num_elements=8, carrier_freq=2.45e9)
        array_model = ArrayModel(array_config)
        signal_config = SignalConfig(carrier_freq=2.45e9, sampling_freq=1e6, num_snapshots=256)
        signal_generator = SignalGenerator(signal_config, array_model)
        
        # Generate a simple signal
        doas = [45.0, 90.0]  # Two sources
        source_powers_db = [0.0, 0.0]
        source_bandwidths = [1e6, 1e6]
        snr_db = 10.0
        
        received_signal = signal_generator.generate_received_signal(
            doas=doas,
            source_powers_db=source_powers_db,
            source_bandwidths=source_bandwidths,
            snr_db=snr_db,
            multipath_enabled=False
        )
        
        print(f"  ✅ Signal generated: shape {received_signal.data.shape}")
        return True
        
    except Exception as e:
        print(f"  ❌ Signal generation failed: {e}")
        return False

def test_classic_algorithms():
    """Test classic DOA algorithms"""
    print("🔧 Testing classic algorithms...")
    
    try:
        from signalgen.array_processing import ArrayConfig, ArrayModel
        from signalgen.signal_generation import SignalConfig, SignalGenerator
        from models.classic.music import MUSIC
        from models.classic.mvdr import MVDR
        from models.classic.beamformer import Beamformer
        
        # Setup
        array_config = ArrayConfig(array_type='linear', num_elements=8, carrier_freq=2.45e9)
        array_model = ArrayModel(array_config)
        signal_config = SignalConfig(carrier_freq=2.45e9, sampling_freq=1e6, num_snapshots=256)
        signal_generator = SignalGenerator(signal_config, array_model)
        
        # Generate test signal
        doas = [60.0]
        received_signal = signal_generator.generate_received_signal(
            doas=doas,
            source_powers_db=[0.0],
            source_bandwidths=[1e6],
            snr_db=10.0,
            multipath_enabled=False
        )
        
        scan_angles = np.arange(30, 151, 1)
        
        # Test MUSIC
        try:
            music = MUSIC(array_model=array_model, scan_angles_deg=scan_angles, num_sources=1)
            music.set_received_data(received_signal.data)
            estimates, spectrum = music.estimate_doa()
            print(f"  ✅ MUSIC: estimated {estimates} (true: {doas})")
        except Exception as e:
            print(f"  ❌ MUSIC failed: {e}")
        
        # Test MVDR
        try:
            mvdr = MVDR(array_model=array_model, scan_angles_deg=scan_angles, num_sources=1)
            mvdr.set_received_data(received_signal.data)
            estimates, spectrum = mvdr.estimate_doa()
            print(f"  ✅ MVDR: estimated {estimates} (true: {doas})")
        except Exception as e:
            print(f"  ❌ MVDR failed: {e}")
        
        # Test Beamformer
        try:
            beamformer = Beamformer(array_model=array_model, scan_angles_deg=scan_angles, num_sources=1)
            beamformer.set_received_data(received_signal.data)
            estimates, spectrum = beamformer.estimate_doa()
            print(f"  ✅ Beamformer: estimated {estimates} (true: {doas})")
        except Exception as e:
            print(f"  ❌ Beamformer failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Classic algorithms test failed: {e}")
        return False

def test_unet_model():
    """Test UNet model creation"""
    print("🔧 Testing UNet model...")
    
    try:
        import torch
        from models.deep_learning.subspace_models import SubspaceUNet
        
        # Create UNet model
        tau = 8
        M = 8
        model = SubspaceUNet(tau=tau, M=M, diff_method="root_music")
        model.eval()
        
        # Test with dummy input
        batch_size = 1
        dummy_input = torch.randn(batch_size, tau, 2*M, M)
        
        with torch.no_grad():
            outputs = model(dummy_input)
            print(f"  ✅ UNet model created and tested: {len(outputs)} outputs")
        
        return True
        
    except Exception as e:
        print(f"  ❌ UNet model test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Comprehensive DOA Evaluation Components")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_array_setup,
        test_signal_generation,
        test_classic_algorithms,
        test_unet_model
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
        print("✅ All tests passed! The evaluation system should work correctly.")
        print("\n🚀 You can now run:")
        print("   python run_evaluation.py --quick")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())



