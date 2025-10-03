"""
RMS Energy Analysis for Cantilever Fatigue Monitoring

This script analyzes the Root Mean Square (RMS) energy of cantilever vibration
data to detect structural fatigue. Changes in RMS energy patterns can indicate
material degradation, crack initiation, and overall structural health changes.

Author: Codeharshw
License: MIT
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RMSAnalyzer:
    """
    Analyzes RMS energy changes in cantilever vibration data.
    
    RMS analysis tracks the overall energy content of the vibration signal.
    Changes in RMS patterns can indicate structural modifications due to
    fatigue, crack propagation, or other forms of material degradation.
    
    Attributes:
        file_path (str): Path to the input data file
        window_size (int): Rolling window size for RMS calculation
        sampling_rate (float): Calculated sampling rate from data
        results_dir (str): Directory for saving results
    """
    
    def __init__(self, file_path='calibrated_mpu9250_data.csv',
                 window_size=200, results_dir='../../results/plots'):
        """
        Initialize the RMSAnalyzer.
        
        Args:
            file_path (str): Path to the CSV data file
            window_size (int): Rolling window size (larger = smoother, smaller = more detail)
            results_dir (str): Directory to save output plots
        """
        self.file_path = file_path
        self.window_size = window_size
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
            df['a_mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
            
            # Calculate sampling rate
            self.sampling_rate = 1 / np.mean(np.diff(df['time']))
            logger.info(f"Calculated sampling rate: {self.sampling_rate:.2f} Hz")
            
            # Store processed data
            self.data = df.copy()
            logger.info(f"Loaded {len(df)} data points spanning {df['time'].iloc[-1]:.2f} seconds")
            
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def calculate_rolling_rms(self) -> np.ndarray:
        """
        Calculate rolling RMS of the acceleration magnitude.
        
        Returns:
            np.ndarray: Rolling RMS values
        """
        logger.info(f"Calculating rolling RMS with window size: {self.window_size}")
        
        # Calculate rolling RMS: sqrt(mean(x^2))
        rms_values = (self.data['a_mag']
                     .rolling(window=self.window_size, min_periods=1)
                     .apply(lambda x: np.sqrt(np.mean(x**2)), raw=True))
        
        # Store in dataframe
        self.data['rms'] = rms_values
        
        # Log statistics
        valid_rms = rms_values.dropna()
        logger.info(f"RMS range: {valid_rms.min():.4f} - {valid_rms.max():.4f} m/s²")
        logger.info(f"Mean RMS: {valid_rms.mean():.4f} ± {valid_rms.std():.4f} m/s²")
        
        return rms_values.values
    
    def calculate_energy_metrics(self) -> Dict[str, float]:
        """
        Calculate additional energy-related metrics.
        
        Returns:
            Dict[str, float]: Dictionary containing energy metrics
        """
        rms_values = self.data['rms'].dropna()
        time_values = self.data['time'][rms_values.index]
        
        metrics = {
            'mean_rms': rms_values.mean(),
            'std_rms': rms_values.std(),
            'min_rms': rms_values.min(),
            'max_rms': rms_values.max(),
            'rms_range': rms_values.max() - rms_values.min(),
            'coefficient_of_variation': rms_values.std() / rms_values.mean(),
            'total_energy': np.trapezoid(rms_values**2, time_values),  # Integrated energy
        }
        
        return metrics
    
    def detect_trend(self) -> Dict[str, Any]:
        """
        Analyze RMS trend over time using linear regression.
        
        Returns:
            Dict[str, Any]: Trend analysis results
        """
        valid_data = self.data[['time', 'rms']].dropna()
        
        if len(valid_data) < 2:
            logger.warning("Insufficient data for trend analysis")
            return {'slope': 0, 'trend': 'insufficient_data'}
        
        # Linear regression
        coefficients = np.polyfit(valid_data['time'], valid_data['rms'], 1)
        slope, intercept = coefficients
        
        # Determine trend
        if abs(slope) < 1e-6:
            trend = 'stable'
        elif slope > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
        
        return {
            'slope': slope,
            'intercept': intercept,
            'trend': trend,
            'slope_per_minute': slope * 60  # Convert to per-minute rate
        }
    
    def plot_results(self, save_plot: bool = True) -> None:
        """
        Create and display RMS analysis plots.
        
        Args:
            save_plot (bool): Whether to save the plot to file
        """
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot 1: RMS vs Time
        ax1.plot(self.data['time'], self.data['rms'], 
                label='Rolling RMS', color='blue', linewidth=2)
        
        # Add statistical overlays
        rms_mean = self.data['rms'].mean()
        rms_std = self.data['rms'].std()
        
        ax1.axhline(y=rms_mean, color='red', linestyle='--', alpha=0.7,
                   label=f'Mean: {rms_mean:.4f} m/s²')
        ax1.fill_between(self.data['time'], 
                        rms_mean - rms_std, rms_mean + rms_std,
                        alpha=0.2, color='red', 
                        label=f'±1σ: {rms_std:.4f} m/s²')
        
        # Add trend line
        trend_info = self.detect_trend()
        if trend_info['trend'] != 'insufficient_data':
            trend_line = (trend_info['slope'] * self.data['time'] + 
                         trend_info['intercept'])
            ax1.plot(self.data['time'], trend_line, 'g--', alpha=0.8,
                    label=f'Trend: {trend_info["trend"]} ({trend_info["slope_per_minute"]:.2e} m/s²/min)')
        
        ax1.set_title('RMS Energy vs. Time - Fatigue Monitoring', 
                     fontsize=16, fontweight='bold')
        ax1.set_ylabel('RMS Acceleration (m/s²)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Raw acceleration magnitude for reference
        ax2.plot(self.data['time'], self.data['a_mag'], 
                color='gray', alpha=0.7, linewidth=0.5,
                label='Raw Acceleration Magnitude')
        ax2.set_title('Raw Acceleration Magnitude (Reference)', fontsize=14)
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Acceleration (m/s²)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add analysis info text box
        metrics = self.calculate_energy_metrics()
        info_text = f'Window Size: {self.window_size} samples\n'
        info_text += f'Sampling Rate: {self.sampling_rate:.1f} Hz\n'
        info_text += f'CV: {metrics["coefficient_of_variation"]:.3f}\n'
        info_text += f'Range: {metrics["rms_range"]:.4f} m/s²'
        
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes,
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.results_dir, 'rms_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {plot_path}")
        
        plt.show()
    
    def run_analysis(self) -> Dict[str, Any]:
        """Run the complete RMS analysis workflow."""
        try:
            logger.info("Starting RMS analysis...")
            
            # Load data
            self.load_data()
            
            # Calculate RMS
            rms_values = self.calculate_rolling_rms()
            
            # Calculate metrics
            metrics = self.calculate_energy_metrics()
            trend_info = self.detect_trend()
            
            # Generate plots
            self.plot_results()
            
            # Prepare results
            results = {
                'time': self.data['time'].values,
                'rms_values': rms_values,
                'raw_acceleration': self.data['a_mag'].values,
                'sampling_rate': self.sampling_rate,
                'window_size': self.window_size,
                'metrics': metrics,
                'trend_analysis': trend_info
            }
            
            logger.info("RMS analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise


def main():
    """Main execution function."""
    # Configuration
    config = {
        'file_path': 'calibrated_mpu9250_data.csv',
        'window_size': 200,  # Rolling window size
        'results_dir': '../../results/plots'
    }
    
    # Run analysis
    analyzer = RMSAnalyzer(**config)
    results = analyzer.run_analysis()
    
    # Print comprehensive summary
    metrics = results['metrics']
    trend = results['trend_analysis']
    
    print("\n" + "="*60)
    print("RMS ENERGY ANALYSIS SUMMARY")
    print("="*60)
    print(f"Sampling Rate: {results['sampling_rate']:.2f} Hz")
    print(f"Window Size: {results['window_size']} samples ({results['window_size']/results['sampling_rate']:.2f} s)")
    print(f"Analysis Duration: {results['time'][-1]:.2f} seconds")
    print("\n" + "-"*40)
    print("ENERGY METRICS:")
    print("-"*40)
    print(f"Mean RMS: {metrics['mean_rms']:.4f} ± {metrics['std_rms']:.4f} m/s²")
    print(f"RMS Range: {metrics['min_rms']:.4f} - {metrics['max_rms']:.4f} m/s²")
    print(f"Coefficient of Variation: {metrics['coefficient_of_variation']:.3f}")
    print(f"Total Energy: {metrics['total_energy']:.2e} (m/s²)²·s")
    print("\n" + "-"*40)
    print("TREND ANALYSIS:")
    print("-"*40)
    print(f"Overall Trend: {trend['trend'].replace('_', ' ').title()}")
    if trend['trend'] != 'insufficient_data':
        print(f"Slope: {trend['slope']:.2e} m/s²/s")
        print(f"Rate of Change: {trend['slope_per_minute']:.2e} m/s²/min")
    print("="*60)


if __name__ == "__main__":
    main()
