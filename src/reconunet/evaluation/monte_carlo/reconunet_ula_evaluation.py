"""
Simple Evaluation Script for MUSIC and MVDR algorithms using Tri4Net ULA Implementation

This script evaluates MUSIC and MVDR DOA estimation algorithms on ULA (Uniform Linear Array)
scenarios with 1, 2, and 3 sources, with controllable angle separation.
Uses Tri4Net components instead of SubspaceNet.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from dataclasses import dataclass

# Add Tri4Net src to path
tri4net_src_path = os.path.join(os.path.dirname(__file__), '..', '..')
if tri4net_src_path not in sys.path:
    sys.path.insert(0, tri4net_src_path)

# Import Tri4Net components
try:
    from signalgen.signal_generator import SignalGenerator, SignalConfig
    from signalgen.array_processing import ArrayModel, ArrayConfig, ReceivedSignal
    from models.classic.music import MUSIC
    from models.classic.mvdr import MVDR
    from models.classic.rootmusic import RootMUSIC
except ImportError as e:
    print(f"Error importing Tri4Net modules: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)


@dataclass
class EvaluationParams:
    """Parameters for evaluation"""
    N: int = 8  # Number of sensors
    T: int = 1024  # Number of snapshots
    snr_range: List[float] = None  # SNR values to test
    num_monte_carlo: int = 25  # Number of Monte Carlo runs
    angle_separations: List[float] = None  # Angle separations to test (degrees)
    scan_range: Tuple[float, float] = (0, 180)  # Scan range in degrees
    scan_resolution: float = 1.0  # Scan resolution in degrees
    carrier_freq: float = 2.45e9  # Carrier frequency in Hz
    fs: float = 1e6  # Sampling frequency in Hz
    
    def __post_init__(self):
        if self.snr_range is None:
            self.snr_range = [-10, -5, 0, 5, 10, 15, 20]
        if self.angle_separations is None:
            self.angle_separations = [3, 5, 10]


class Tri4NetULAEvaluator:
    """Simple evaluator for MUSIC and MVDR using Tri4Net ULA"""
    
    def __init__(self, params: EvaluationParams):
        self.params = params
        self.scan_angles = np.arange(
            params.scan_range[0], 
            params.scan_range[1] + params.scan_resolution, 
            params.scan_resolution
        )
        
        # Create array configuration for ULA
        self.array_config = ArrayConfig(
            array_type='linear',
            num_elements=params.N,
            carrier_freq=params.carrier_freq,
            element_spacing=0.5,  # Half wavelength spacing
            enable_gain_phase_errors=False,  # Disable for fair comparison
            enable_mutual_coupling=False,    # Disable for fair comparison
            position_error_std=0.0           # No position errors
        )
        
    def generate_test_angles(self, num_sources: int, angle_separation: float) -> List[float]:
        """Generate test angles with specified separation"""
        if num_sources == 1:
            return [90.0]  # Single source at broadside
        elif num_sources == 2:
            return [90.0, 90.0 + angle_separation]
        elif num_sources == 3:
            return [90.0 - angle_separation, 90.0, 90.0 + angle_separation]
        else:
            raise ValueError(f"Unsupported number of sources: {num_sources}")
    
    def create_test_signal(self, true_angles: List[float], snr: float) -> Tuple[np.ndarray, ArrayModel]:
        """Create test signal with given angles and SNR using Tri4Net signal generation"""
        # Create signal configuration
        signal_config = SignalConfig(
            fs=self.params.fs,
            T=self.params.T,
            num_sources=len(true_angles),
            angles=true_angles,
            frequencies=[1e5] * len(true_angles),  # Fixed frequency for narrowband
            snr_db=snr,
            main_source_power=0.0,  # 0 dB reference power
            use_full_bandwidth=True
        )
        
        # Generate source signals using SignalGenerator
        signal_generator = SignalGenerator(signal_config)
        time_axis, source_signals = signal_generator.generate_signals()
        
        # Create array model
        array_model = ArrayModel(self.array_config)
        
        # Create steering matrix for the true angles
        steering_matrix = array_model.steering_matrix(true_angles)
        
        # Create received signal object with source signals and steering matrix
        received_signal = ReceivedSignal(
            steering_matrix=steering_matrix,
            source_signals=source_signals,
            source_angles=true_angles,
            signal_config=signal_config
        )
        
        # Add noise to achieve desired SNR
        noisy_received_signal = received_signal.add_noise(
            snr_db=snr,
            signal_bandwidth=signal_config.bandwidth,
            sampling_frequency=signal_config.fs,
            source_frequencies=signal_config.frequencies,
        )
        
        # Get the final array signals with noise
        final_array_signals = noisy_received_signal.array_signals
        
        return final_array_signals, array_model
    
    def estimate_doa_music(self, X: np.ndarray, array_model: ArrayModel, num_sources: int) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate DOA using MUSIC algorithm"""
        music = MUSIC(
            array_model=array_model,
            scan_angles_deg=self.scan_angles,
            num_sources=num_sources
        )
        
        # Set received data and estimate DOAs
        music.set_received_data(X)
        estimated_angles, spectrum = music.estimate_doa()
        
        return estimated_angles, spectrum
    
    def estimate_doa_mvdr(self, X: np.ndarray, array_model: ArrayModel, num_sources: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate DOA using MVDR algorithm"""
        mvdr = MVDR(
            array_model=array_model,
            scan_angles_deg=self.scan_angles,
            num_sources=num_sources
        )
        
        # Set received data and estimate DOAs
        mvdr.set_received_data(X)
        estimated_angles, spectrum = mvdr.estimate_doa()
        
        return estimated_angles, spectrum
    
    def estimate_doa_root_music(self, X: np.ndarray, array_model: ArrayModel, num_sources: int) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate DOA using Root MUSIC algorithm"""
        root_music = RootMUSIC(
            array_model=array_model,
            scan_angles_deg=self.scan_angles,
            num_sources=num_sources
        )
        
        # Set received data and estimate DOAs
        root_music.set_received_data(X)
        
        # RootMUSIC returns 5 values: (doa_predictions, roots, doa_predictions_all, roots_angels_all, M)
        # We only need the first one for estimated angles
        estimated_angles, roots, doa_predictions_all, roots_angels_all, M = root_music.estimate_doa()
        
        
        return estimated_angles
    
    def calculate_rmse(self, true_angles: List[float], estimated_angles: np.ndarray) -> float:
        """Calculate RMSE between true and estimated angles"""
        if len(estimated_angles) != len(true_angles):
            return float('inf')  # Penalty for wrong number of sources
        
        # Check for NaN values
        if np.any(np.isnan(estimated_angles)):
            return float('inf')
        
        # Convert to numpy arrays for consistency
        true_angles = np.array(true_angles)
        estimated_angles = np.array(estimated_angles)
        
        # Find best permutation match
        min_error = float('inf')
        from itertools import permutations
        
        for perm in permutations(estimated_angles):
            perm_array = np.array(perm)
            if not np.any(np.isnan(perm_array)):
                error = np.sqrt(np.mean((true_angles - perm_array)**2))
                min_error = min(min_error, error)
        
        return min_error
    
    def run_single_scenario(self, num_sources: int, angle_separation: float, snr: float, 
                          algorithm: str = 'MUSIC') -> Dict:
        """Run evaluation for a single scenario"""
        true_angles = self.generate_test_angles(num_sources, angle_separation)
        
        rmse_values = []
        success_count = 0
        
        for _ in range(self.params.num_monte_carlo):
            try:
                # Generate test signal
                X, array_model = self.create_test_signal(true_angles, snr)
                
                # Estimate DOAs
                if algorithm.upper() == 'MUSIC':
                    estimated_angles, _ = self.estimate_doa_music(X, array_model, num_sources)
                elif algorithm.upper() == 'MVDR':
                    estimated_angles, _ = self.estimate_doa_mvdr(X, array_model, num_sources)
                elif algorithm.upper() == 'ROOT_MUSIC' or algorithm.upper() == 'ROOT-MUSIC':
                    estimated_angles = self.estimate_doa_root_music(X, array_model, num_sources)
                else:
                    raise ValueError(f"Unknown algorithm: {algorithm}")
                
                # Ensure estimated_angles is a numpy array and take only the required number
                estimated_angles = np.array(estimated_angles).flatten()
                
                # Take only the first num_sources estimates
                if len(estimated_angles) >= num_sources:
                    estimated_subset = estimated_angles[:num_sources]
                else:
                    # If not enough estimates, pad with NaN or return inf RMSE
                    estimated_subset = np.full(num_sources, np.nan)
                
                # Calculate RMSE
                rmse = self.calculate_rmse(true_angles, estimated_subset)
                rmse_values.append(rmse)
                
                # Count successful estimations (RMSE < 5 degrees)
                if not np.isnan(rmse) and not np.isinf(rmse) and rmse < 5.0:
                    success_count += 1
                    
            except Exception as e:
                print(f"Error in Monte Carlo run: {e}")
                rmse_values.append(float('inf'))
        
        # Calculate statistics
        valid_rmse = [r for r in rmse_values if r != float('inf')]
        
        return {
            'num_sources': num_sources,
            'angle_separation': angle_separation,
            'snr': snr,
            'algorithm': algorithm,
            'mean_rmse': np.mean(valid_rmse) if valid_rmse else float('inf'),
            'std_rmse': np.std(valid_rmse) if valid_rmse else 0,
            'success_rate': success_count / self.params.num_monte_carlo,
            'num_trials': self.params.num_monte_carlo,
            'true_angles': true_angles
        }
    
    def run_comprehensive_evaluation(self) -> Dict:
        """Run comprehensive evaluation for all scenarios"""
        results = {
            'MUSIC': [],
            'MVDR': [],
            'ROOT_MUSIC': []
        }
        
        total_scenarios = len([1, 2, 3]) * len(self.params.angle_separations) * len(self.params.snr_range) * 3
        scenario_count = 0
        
        for algorithm in ['MUSIC', 'MVDR', 'ROOT_MUSIC']:
            for num_sources in [1, 2, 3]:
                for angle_separation in self.params.angle_separations:
                    # Skip angle separation for single source
                    if num_sources == 1 and angle_separation != self.params.angle_separations[0]:
                        continue
                        
                    for snr in self.params.snr_range:
                        scenario_count += 1
                        print(f"Running scenario {scenario_count}/{total_scenarios}: "
                              f"{algorithm}, {num_sources} sources, {angle_separation}° separation, "
                              f"{snr} dB SNR")
                        
                        result = self.run_single_scenario(num_sources, angle_separation, snr, algorithm)
                        results[algorithm].append(result)
        
        return results
    
    def plot_results(self, results: Dict, save_path: str = None):
        """Plot evaluation results - one figure per angle separation"""
        
        # Get all unique angle separations from results
        all_separations = set()
        for algorithm in ['MUSIC', 'MVDR', 'ROOT_MUSIC']:
            for result in results[algorithm]:
                all_separations.add(result['angle_separation'])
        
        all_separations = sorted(list(all_separations))
        
        # Create one figure for each angle separation
        for separation in all_separations:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            fig.suptitle(f'MUSIC vs MVDR vs Root-MUSIC Performance (Tri4Net) - {separation}° Angle Separation', fontsize=14)
            
            # Plot all three algorithms
            for algorithm in ['MUSIC', 'MVDR', 'ROOT_MUSIC']:
                alg_results = results[algorithm]
                
                # Plot for different number of sources
                for num_sources in [1, 2, 3]:
                    snr_vals = []
                    rmse_vals = []
                    
                    # Collect data for this algorithm, num_sources, and separation
                    for result in alg_results:
                        if (result['num_sources'] == num_sources and 
                            result['angle_separation'] == separation):
                            snr_vals.append(result['snr'])
                            rmse_vals.append(result['mean_rmse'])
                    
                    # Sort by SNR for proper line plotting
                    if snr_vals:
                        sorted_data = sorted(zip(snr_vals, rmse_vals))
                        snr_vals, rmse_vals = zip(*sorted_data)
                        
                        # Choose line style and marker
                        if algorithm == 'MUSIC':
                            linestyle = '-'
                            marker = 'o'
                        elif algorithm == 'MVDR':
                            linestyle = '--'
                            marker = 's'
                        else:  # ROOT_MUSIC
                            linestyle = ':'
                            marker = '^'
                        
                        ax.plot(snr_vals, rmse_vals, 
                               linestyle=linestyle, marker=marker, 
                               label=f'{algorithm} - {num_sources} source(s)',
                               linewidth=2, markersize=6)
            
            ax.set_xlabel('SNR (dB)', fontsize=12)
            ax.set_ylabel('RMSE (degrees)', fontsize=12)
            ax.set_title(f'RMSE vs SNR for {separation}° Separation', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Set y-axis to log scale if there are large variations
            # y_vals = [line.get_ydata() for line in ax.lines]
            # if y_vals:
            #     all_y_vals = [y for line_y in y_vals for y in line_y if not np.isnan(y) and not np.isinf(y)]
            #     if all_y_vals and max(all_y_vals) / min(all_y_vals) > 100:
            #         ax.set_yscale('log')
            
            plt.tight_layout()
            
            # Save individual plots if save_path is provided
            if save_path:
                base_path = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
                sep_save_path = f"{base_path}_separation_{separation}deg.png"
                plt.savefig(sep_save_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {sep_save_path}")
            
            plt.show()
        
        print(f"Generated {len(all_separations)} plots for angle separations: {all_separations}°")
    
    def save_results(self, results: Dict, save_path: str):
        """Save results to file"""
        import json
        
        def convert_numpy_types(obj):
            """Convert numpy types to native Python types for JSON serialization"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert numpy types to native Python types for JSON serialization
        json_results = convert_numpy_types(results)
        
        with open(save_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to {save_path}")


def main():
    """Main evaluation function"""
    # Set evaluation parameters (same as SubspaceNet version)
    eval_params = EvaluationParams(
        N=8,  # 8-element ULA
        T=1024,  # 1024 snapshots
        snr_range=np.arange(-10, 16, 1),
        num_monte_carlo=100,  # Reduced for faster execution
        angle_separations=[3, 5, 10, 20],
        scan_resolution=1.0,
        carrier_freq=2.45e9,  # 2.45 GHz
        fs=1e3  # 1 MHz sampling frequency
    )
    
    # Create evaluator
    evaluator = Tri4NetULAEvaluator(eval_params)
    
    print("Starting Tri4Net ULA Evaluation...")
    print(f"Parameters: {eval_params.N} sensors, {eval_params.T} snapshots, {eval_params.num_monte_carlo} MC runs")
    
    # Run evaluation
    results = evaluator.run_comprehensive_evaluation()
    
    # Plot results (will generate one plot per angle separation)
    evaluator.plot_results(results, 'tri4net_ula_evaluation')
    
    # Save results
    evaluator.save_results(results, 'tri4net_ula_evaluation_results.json')
    
    # Print summary
    print("\nEvaluation Summary:")
    for algorithm in ['MUSIC', 'MVDR', 'ROOT_MUSIC']:
        alg_results = results[algorithm]
        avg_success = np.mean([r['success_rate'] for r in alg_results])
        print(f"{algorithm}: Average success rate = {avg_success:.3f}")


if __name__ == "__main__":
    main() 