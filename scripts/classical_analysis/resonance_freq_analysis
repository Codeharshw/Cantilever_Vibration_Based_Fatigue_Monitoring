#!/usr/bin/env python3
"""
Resonance Frequency Analysis for Cantilever Fatigue Monitoring

This script analyzes the resonance frequency shifts in cantilever vibration data
to detect structural fatigue. As fatigue progresses, the cantilever's stiffness
decreases, leading to a measurable reduction in natural frequency.

Author: [Your Name]
Date: [Date]
License: MIT
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FrequencyAnalyzer:
    """
    Analyzes resonance frequency changes in cantilever vibration data.
    
    Attributes:
        file_path (str): Path to the input data file
        nperseg (int): STFT window size for frequency analysis
        sampling_rate (float): Calculated sampling rate from data
        results_dir (str): Directory for saving results
    """
    
    def __init__(self, file_path='../../data/calibrated_mpu9250_data.csv', nperseg=256, 
                 results_dir='../../results/plots'):
        """
        Initialize the FrequencyAnalyzer.
        
        Args:
            file_path (str): Path to the CSV data file
            nperseg (int): STFT window size (affects frequency/time resolution trade-off)
            results_dir (str): Directory to save output plots
        """
        self.file_path = file_path
        self.nperseg = nperseg
        self.results_dir = results_dir
        self.sampling_rate = None
        self.data = None
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
    
    def load_data(self):
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
    
    def calculate_stft(self):
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
    
    def find_resonance_frequencies(self, f, Zxx):
        """
        Extract dominant frequencies from STFT data.
        
        Args:
            f (ndarray): Frequency array from STFT
            Zxx (ndarray): STFT matrix
            
        Returns:
            ndarray: Array of dominant frequencies over time
        """
        # Find frequency with maximum power for each time segment
        resonance_freqs = f[np.argmax(np.abs(Zxx), axis=0)]
        return resonance_freqs
    
    def plot_results(self, f, t, Zxx, resonance_freqs, save_plot=True):
        """
        Create and display frequency analysis plots.
        
        Args:
            f (ndarray): Frequency array
            t (ndarray): Time array
            Zxx (ndarray): STFT matrix
            resonance_freqs (ndarray): Dominant frequencies over time
            save_plot (bool): Whether to save the plot to file
        """
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot 1: Spectrogram
        freq_limit = self.sampling_rate / 4  # Nyquist frequency / 2 for better visualization
        
        spec = ax1.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap='viridis')
        fig.colorbar(spec, ax=ax1, label='Magnitude')
        ax1.set_title('Acceleration Spectrogram', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Frequency (Hz)', fontsize=12)
        ax1.set_ylim(0, freq_limit)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Resonance frequency over time
        ax2.plot(t, resonance_freqs, 'r.-', linewidth=2, markersize=4, 
                label='Dominant Frequency')
        ax2.set_title('Resonance Frequency vs. Time', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Frequency (Hz)', fontsize=12)
        ax2.set_ylim(0, freq_limit)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add statistical information
        mean_freq = np.mean(resonance_freqs)
        std_freq = np.std(resonance_freqs)
        ax2.axhline(y=mean_freq, color='blue', linestyle='--', alpha=0.7, 
                   label=f'Mean: {mean_freq:.2f} Hz')
        ax2.fill_between(t, mean_freq - std_freq, mean_freq + std_freq, 
                        alpha=0.2, color='blue', label=f'±1σ: {std_freq:.2f} Hz')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.results_dir, 'frequency_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {plot_path}")
        
        plt.show()
    
    def run_analysis(self):
        """Run the complete frequency analysis workflow."""
        try:
            logger.info("Starting frequency analysis...")
            
            # Load data
            self.load_data()
            
            # Calculate STFT
            f, t, Zxx = self.calculate_stft()
            
            # Find resonance frequencies
            resonance_freqs = self.find_resonance_frequencies(f, Zxx)
            
            # Generate plots
            self.plot_results(f, t, Zxx, resonance_freqs)
            
            # Log results summary
            logger.info(f"Analysis complete. Frequency range: {resonance_freqs.min():.2f} - {resonance_freqs.max():.2f} Hz")
            logger.info(f"Mean frequency: {np.mean(resonance_freqs):.2f} ± {np.std(resonance_freqs):.2f} Hz")
            
            return {
                'frequencies': f,
                'time': t,
                'stft_matrix': Zxx,
                'resonance_frequencies': resonance_freqs,
                'sampling_rate': self.sampling_rate
            }
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise


def main():
    """Main execution function."""
    # Configuration
    config = {
        'file_path': 'calibrated_mpu9250_data.csv', # Upload your accelerometer CSV file here
        'nperseg': 256,  # STFT window size
        'results_dir': '../../results/plots'
    }
    
    # Run analysis
    analyzer = FrequencyAnalyzer(**config)
    results = analyzer.run_analysis()
    
    print("\n" + "="*50)
    print("FREQUENCY ANALYSIS SUMMARY")
    print("="*50)
    print(f"Sampling Rate: {results['sampling_rate']:.2f} Hz")
    print(f"Frequency Range: {results['resonance_frequencies'].min():.2f} - {results['resonance_frequencies'].max():.2f} Hz")
    print(f"Mean Resonance Frequency: {np.mean(results['resonance_frequencies']):.2f} ± {np.std(results['resonance_frequencies']):.2f} Hz")
    print(f"Analysis Duration: {results['time'][-1]:.2f} seconds")
    print("="*50)


if __name__ == "__main__":
    main()
