"""
Peak Width (Damping) Analysis for Cantilever Fatigue Monitoring

This script analyzes the Full Width at Half Maximum (FWHM) of resonance peaks
to detect structural fatigue. As fatigue progresses and cracks develop, the
cantilever's damping increases, resulting in broader resonance peaks.

Author: Codeharshw
License: MIT
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import logging
from typing import Tuple, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DampingAnalyzer:
    """
    Analyzes peak width changes (damping) in cantilever vibration data.
    
    Peak width analysis uses Full Width at Half Maximum (FWHM) to quantify
    the damping characteristics of the system. Increased damping from crack
    formation results in broader resonance peaks.
    
    Attributes:
        file_path (str): Path to the input data file
        nperseg (int): STFT window size for frequency analysis
        sampling_rate (float): Calculated sampling rate from data
        results_dir (str): Directory for saving results
    """
    
    def __init__(self, file_path='calibrated_mpu9250_data.csv', # Upload your accelerometer data here
                 nperseg=512, results_dir='../../results/plots'):
        """
        Initialize the DampingAnalyzer.
        
        Args:
            file_path (str): Path to the CSV data file
            nperseg (int): STFT window size (larger = better frequency resolution)
            results_dir (str): Directory to save output plots
        """
        self.file_path = file_path
        self.nperseg = nperseg
        self.results_dir = results_dir
        self.sampling_rate = None
        self.data = None
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
    
    def load_data(self) -> None:
        """Load and preprocess the acceleration data."""
        try:
            logger.info(f"Loading data from {self.file_path}")
            df = pd.read_csv(self.file_path, header=None)
            df.columns = ['time', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
            
            # Calculate acceleration magnitude
            accel_mag = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2).values
            time_vec = df['time'].values
            
            # Calculate sampling rate
            self.sampling_rate = 1 / np.mean(np.diff(time_vec))
            logger.info(f"Calculated sampling rate: {self.sampling_rate:.2f} Hz")
            
            self.data = {
                'time': time_vec,
                'accel_mag': accel_mag
            }
            
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def calculate_stft(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Short-Time Fourier Transform of the acceleration data.
        
        Returns:
            tuple: Frequency array, time array, and STFT matrix
        """
        logger.info("Calculating STFT...")
        f, t, Zxx = signal.stft(
            self.data['accel_mag'], 
            fs=self.sampling_rate, 
            nperseg=self.nperseg
        )
        return f, t, Zxx
    
    def calculate_fwhm(self, spectrum_slice: np.ndarray, 
                      frequencies: np.ndarray) -> Optional[float]:
        """
        Calculate Full Width at Half Maximum for a spectrum slice.
        
        Args:
            spectrum_slice (np.ndarray): Magnitude spectrum for one time slice
            frequencies (np.ndarray): Frequency array corresponding to spectrum
            
        Returns:
            Optional[float]: FWHM width in Hz, or None if calculation fails
        """
        try:
            peak_idx = np.argmax(spectrum_slice)
            peak_mag = spectrum_slice[peak_idx]
            half_max = peak_mag / 2.0
            
            # Find crossing points on left and right sides of peak
            left_side = np.where(spectrum_slice[:peak_idx] < half_max)[0]
            right_side = np.where(spectrum_slice[peak_idx:] < half_max)[0] + peak_idx
            
            # Get indices closest to peak
            left_idx = left_side[-1] if len(left_side) > 0 else 0
            right_idx = right_side[0] if len(right_side) > 0 else len(spectrum_slice) - 1
            
            # Calculate width in Hz
            width = frequencies[right_idx] - frequencies[left_idx]
            return width
            
        except (IndexError, ValueError):
            return None
    
    def calculate_peak_widths(self, frequencies: np.ndarray, 
                            magnitudes: np.ndarray) -> List[float]:
        """
        Calculate peak widths for all time slices.
        
        Args:
            frequencies (np.ndarray): Frequency array
            magnitudes (np.ndarray): STFT magnitude matrix
            
        Returns:
            List[float]: Peak widths over time (NaN for failed calculations)
        """
        logger.info("Calculating peak widths (FWHM)...")
        peak_widths = []
        
        for i in range(magnitudes.shape[1]):
            spectrum_slice = magnitudes[:, i]
            width = self.calculate_fwhm(spectrum_slice, frequencies)
            peak_widths.append(width if width is not None else np.nan)
        
        # Log statistics
        valid_widths = [w for w in peak_widths if not np.isnan(w)]
        if valid_widths:
            logger.info(f"Successfully calculated {len(valid_widths)}/{len(peak_widths)} peak widths")
            logger.info(f"Width range: {min(valid_widths):.3f} - {max(valid_widths):.3f} Hz")
        else:
            logger.warning("No valid peak widths calculated")
        
        return peak_widths
    
    def plot_results(self, time_array: np.ndarray, peak_widths: List[float], 
                    save_plot: bool = True) -> None:
        """
        Create and display damping analysis plots.
        
        Args:
            time_array (np.ndarray): Time array from STFT
            peak_widths (List[float]): Peak width values over time
            save_plot (bool): Whether to save the plot to file
        """
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Convert to numpy array for easier handling
        peak_widths_array = np.array(peak_widths)
        
        # Plot peak widths
        ax.plot(time_array, peak_widths_array, marker='.', linestyle='-', 
                color='purple', linewidth=2, markersize=4, 
                label='Peak Width (FWHM)')
        
        # Add statistical information
        valid_mask = ~np.isnan(peak_widths_array)
        if np.any(valid_mask):
            valid_widths = peak_widths_array[valid_mask]
            mean_width = np.mean(valid_widths)
            std_width = np.std(valid_widths)
            
            ax.axhline(y=mean_width, color='red', linestyle='--', alpha=0.7,
                      label=f'Mean: {mean_width:.3f} Hz')
            ax.fill_between(time_array, mean_width - std_width, mean_width + std_width,
                           alpha=0.2, color='red', label=f'±1σ: {std_width:.3f} Hz')
        
        # Formatting
        ax.set_title('Peak Width (FWHM) vs. Time - Damping Analysis', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Frequency Width (Hz)', fontsize=12)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add text box with analysis info
        info_text = f'Window Size: {self.nperseg} samples\n'
        info_text += f'Sampling Rate: {self.sampling_rate:.1f} Hz\n'
        if np.any(valid_mask):
            info_text += f'Valid Measurements: {np.sum(valid_mask)}/{len(peak_widths)}'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.results_dir, 'damping_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {plot_path}")
        
        plt.show()
    
    def run_analysis(self) -> dict:
        """Run the complete damping analysis workflow."""
        try:
            logger.info("Starting damping analysis...")
            
            # Load data
            self.load_data()
            
            # Calculate STFT
            f, t, Zxx = self.calculate_stft()
            magnitudes = np.abs(Zxx)
            
            # Calculate peak widths
            peak_widths = self.calculate_peak_widths(f, magnitudes)
            
            # Generate plots
            self.plot_results(t, peak_widths)
            
            # Prepare results
            results = {
                'time': t,
                'peak_widths': np.array(peak_widths),
                'frequencies': f,
                'magnitudes': magnitudes,
                'sampling_rate': self.sampling_rate
            }
            
            # Log summary statistics
            valid_widths = [w for w in peak_widths if not np.isnan(w)]
            if valid_widths:
                logger.info(f"Analysis complete. Mean peak width: {np.mean(valid_widths):.3f} ± {np.std(valid_widths):.3f} Hz")
            
            return results
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise


def main():
    """Main execution function."""
    # Configuration
    config = {
        'file_path': 'calibrated_mpu9250_data.csv',
        'nperseg': 512,  # Larger window for better frequency resolution
        'results_dir': '../../results/plots'
    }
    
    # Run analysis
    analyzer = DampingAnalyzer(**config)
    results = analyzer.run_analysis()
    
    # Print summary
    peak_widths = results['peak_widths']
    valid_mask = ~np.isnan(peak_widths)
    valid_widths = peak_widths[valid_mask]
    
    print("\n" + "="*50)
    print("DAMPING ANALYSIS SUMMARY")
    print("="*50)
    print(f"Sampling Rate: {results['sampling_rate']:.2f} Hz")
    print(f"STFT Window Size: {config['nperseg']} samples")
    print(f"Valid Measurements: {len(valid_widths)}/{len(peak_widths)}")
    
    if len(valid_widths) > 0:
        print(f"Peak Width Range: {valid_widths.min():.3f} - {valid_widths.max():.3f} Hz")
        print(f"Mean Peak Width: {valid_widths.mean():.3f} ± {valid_widths.std():.3f} Hz")
        print(f"Analysis Duration: {results['time'][-1]:.2f} seconds")
    else:
        print("No valid peak width measurements obtained")
    
    print("="*50)


if __name__ == "__main__":
    main()
