"""
IMU Data FFT Analysis Tool
==========================

A comprehensive tool for analyzing IMU (Inertial Measurement Unit) sensor data
using Fast Fourier Transform (FFT) to identify frequency domain characteristics.

Features:
- Supports 6-axis IMU data (3-axis accelerometer + 3-axis gyroscope)
- Comprehensive frequency domain analysis
- Statistical summaries and dominant frequency detection
- Professional visualization with multiple plot types
- Compatible with MPU9250 and similar IMU sensors

Author: [Codeharshw]
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

# Configure matplotlib for professional output
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9

class IMUAnalyzer:
    """
    A class for analyzing IMU sensor data using FFT analysis.
    """
    
    def __init__(self, csv_file_path):
        """
        Initialize the IMU analyzer with data from CSV file.
        
        Args:
            csv_file_path (str): Path to the CSV file containing IMU data
        """
        self.csv_path = csv_file_path
        self.df = None
        self.fft_results = {}
        self.sensor_config = {
            'ax': {'label': 'Accel X', 'unit': 'g', 'color': '#e74c3c', 'type': 'accel'},
            'ay': {'label': 'Accel Y', 'unit': 'g', 'color': '#3498db', 'type': 'accel'},
            'az': {'label': 'Accel Z', 'unit': 'g', 'color': '#2ecc71', 'type': 'accel'},
            'gx': {'label': 'Gyro X', 'unit': 'deg/s', 'color': '#f39c12', 'type': 'gyro'},
            'gy': {'label': 'Gyro Y', 'unit': 'deg/s', 'color': '#9b59b6', 'type': 'gyro'},
            'gz': {'label': 'Gyro Z', 'unit': 'deg/s', 'color': '#1abc9c', 'type': 'gyro'}
        }
        
    def load_data(self):
        """Load IMU data from CSV file."""
        try:
            self.df = pd.read_csv(self.csv_path)
            self.df.columns = ['time', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
            print(f"Successfully loaded data: {self.df.shape[0]} samples")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def calculate_sampling_params(self):
        """Calculate sampling parameters from the data."""
        t = self.df['time'].values
        dt = np.mean(np.diff(t))
        fs = 1 / dt
        n_samples = len(self.df)
        duration = self.df['time'].iloc[-1] - self.df['time'].iloc[0]
        
        self.sampling_params = {
            'dt': dt,
            'fs': fs,
            'n_samples': n_samples,
            'duration': duration,
            'nyquist_freq': fs / 2
        }
        
        return self.sampling_params
    
    def perform_fft_analysis(self):
        """Perform FFT analysis on all sensor channels."""
        print("Performing FFT analysis...")
        
        for channel, config in self.sensor_config.items():
            signal = self.df[channel].values
            
            # Apply FFT
            fft_vals = np.fft.fft(signal)
            fft_freqs = np.fft.fftfreq(self.sampling_params['n_samples'], 
                                     d=self.sampling_params['dt'])
            
            # Keep only positive frequencies
            pos_mask = fft_freqs >= 0
            fft_vals_pos = fft_vals[pos_mask]
            fft_freqs_pos = fft_freqs[pos_mask]
            
            # Calculate signal statistics
            signal_stats = {
                'mean': np.mean(signal),
                'std': np.std(signal),
                'min': np.min(signal),
                'max': np.max(signal),
                'peak_to_peak': np.max(signal) - np.min(signal)
            }
            
            # Store results
            self.fft_results[channel] = {
                'frequencies': fft_freqs_pos,
                'amplitudes': np.abs(fft_vals_pos),
                'phases': np.angle(fft_vals_pos),
                'signal_stats': signal_stats
            }
        
        print("FFT analysis completed successfully")
    
    def find_dominant_frequencies(self):
        """Find dominant frequencies for each channel."""
        dominant_freqs = {}
        
        for channel, fft_data in self.fft_results.items():
            freqs = fft_data['frequencies']
            amps = fft_data['amplitudes']
            
            # Exclude DC component and very low frequencies
            non_dc_mask = freqs > 0.1
            
            if np.any(non_dc_mask):
                peak_indices = np.where(non_dc_mask)[0]
                if len(peak_indices) > 0:
                    max_amp_idx = peak_indices[np.argmax(amps[peak_indices])]
                    dominant_freqs[channel] = {
                        'frequency': freqs[max_amp_idx],
                        'amplitude': amps[max_amp_idx]
                    }
                else:
                    dominant_freqs[channel] = {'frequency': 0, 'amplitude': 0}
            else:
                dominant_freqs[channel] = {'frequency': 0, 'amplitude': 0}
        
        return dominant_freqs
    
    def create_comprehensive_plots(self, save_path=None):
        """Create comprehensive visualization plots."""
        fig = plt.figure(figsize=(16, 28))  # Increased height for 6 individual plots
        
        # Time domain plots
        plt.subplot(6, 2, 1)
        for channel, config in self.sensor_config.items():
            if config['type'] == 'accel':
                plt.plot(self.df['time'], self.df[channel], 
                        label=config['label'], color=config['color'], linewidth=1.5)
        plt.title('Accelerometer Data - Time Domain', fontweight='bold')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (g)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(6, 2, 2)
        for channel, config in self.sensor_config.items():
            if config['type'] == 'gyro':
                plt.plot(self.df['time'], self.df[channel], 
                        label=config['label'], color=config['color'], linewidth=1.5)
        plt.title('Gyroscope Data - Time Domain', fontweight='bold')
        plt.xlabel('Time (s)')
        plt.ylabel('Angular Velocity (deg/s)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Frequency domain plots
        plt.subplot(6, 2, 3)
        for channel, config in self.sensor_config.items():
            if config['type'] == 'accel':
                fft_data = self.fft_results[channel]
                plt.plot(fft_data['frequencies'], fft_data['amplitudes'],
                        label=f"{config['label']} FFT", color=config['color'], linewidth=1.5)
        plt.title('Accelerometer FFT - Frequency Domain', fontweight='bold')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(-0.2, self.sampling_params['nyquist_freq'])
        
        plt.subplot(6, 2, 4)
        for channel, config in self.sensor_config.items():
            if config['type'] == 'gyro':
                fft_data = self.fft_results[channel]
                plt.plot(fft_data['frequencies'], fft_data['amplitudes'],
                        label=f"{config['label']} FFT", color=config['color'], linewidth=1.5)
        plt.title('Gyroscope FFT - Frequency Domain', fontweight='bold')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(-0.2, self.sampling_params['nyquist_freq'])
        
        # Individual channel FFT plots (all 6 channels)
        subplot_positions = [(6, 2, 5), (6, 2, 6), (6, 2, 7), (6, 2, 8), (6, 2, 9), (6, 2, 10)]
        channels_to_plot = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
        
        for i, channel in enumerate(channels_to_plot):
            plt.subplot(*subplot_positions[i])
            config = self.sensor_config[channel]
            fft_data = self.fft_results[channel]
            
            plt.plot(fft_data['frequencies'], fft_data['amplitudes'],
                    color=config['color'], linewidth=2)
            plt.title(f'{config["label"]} - Detailed FFT', fontweight='bold')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)
            plt.xlim(-0.2, min(50, self.sampling_params['nyquist_freq']))
        
        plt.tight_layout(pad=3.0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plots saved to: {save_path}")
        
        plt.show()
    
    def generate_report(self):
        """Generate a comprehensive analysis report."""
        print("\n" + "="*60)
        print("IMU DATA FFT ANALYSIS REPORT")
        print("="*60)
        
        # Dataset information
        print(f"\nDATASET INFORMATION:")
        print(f"File: {Path(self.csv_path).name}")
        print(f"Shape: {self.df.shape}")
        print(f"Duration: {self.sampling_params['duration']:.3f} seconds")
        print(f"Sampling rate: {self.sampling_params['fs']:.1f} Hz")
        print(f"Nyquist frequency: {self.sampling_params['nyquist_freq']:.2f} Hz")
        
        # Dominant frequencies
        print(f"\nDOMINANT FREQUENCY ANALYSIS:")
        print("-" * 40)
        dominant_freqs = self.find_dominant_frequencies()
        
        for channel, config in self.sensor_config.items():
            dom_data = dominant_freqs[channel]
            stats = self.fft_results[channel]['signal_stats']
            
            print(f"{config['label']} ({channel}):")
            print(f"  Dominant frequency: {dom_data['frequency']:.2f} Hz")
            print(f"  Amplitude: {dom_data['amplitude']:.4f}")
            print(f"  Signal range: {stats['min']:.3f} to {stats['max']:.3f} {config['unit']}")
            print(f"  Standard deviation: {stats['std']:.3f} {config['unit']}")
            print()
        
        # Summary statistics table
        print("SUMMARY STATISTICS TABLE:")
        print("-" * 40)
        
        summary_data = []
        for channel, config in self.sensor_config.items():
            stats = self.fft_results[channel]['signal_stats']
            dom_data = dominant_freqs[channel]
            
            summary_data.append({
                'Channel': f"{config['label']} ({channel})",
                'Mean': f"{stats['mean']:.3f}",
                'Std': f"{stats['std']:.3f}",
                'Range': f"{stats['peak_to_peak']:.3f}",
                'Dom_Freq_Hz': f"{dom_data['frequency']:.2f}",
                'Dom_Amplitude': f"{dom_data['amplitude']:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        print(f"\nANALYSIS NOTES:")
        print(f"- Frequency resolution: {self.fft_results['ax']['frequencies'][1]:.3f} Hz")
        print(f"- Analysis based on calibrated IMU sensor data")
        print(f"- Low dominant frequencies may indicate slow movements or drift")
        print(f"- Consider high-pass filtering for motion analysis applications")
        
        return summary_df


def main():
    """Main function to run the IMU analysis."""
    # Configuration
    CSV_FILE_PATH = "your_accelerometer_data.csv"  # Upload your accelerometer CSV file here
    SAVE_PLOTS = True
    PLOTS_OUTPUT_PATH = "imu_fft_analysis_plots.png"
    
    # Initialize analyzer
    analyzer = IMUAnalyzer(CSV_FILE_PATH)
    
    # Load and analyze data
    if analyzer.load_data():
        analyzer.calculate_sampling_params()
        analyzer.perform_fft_analysis()
        
        # Generate visualizations
        if SAVE_PLOTS:
            analyzer.create_comprehensive_plots(save_path=PLOTS_OUTPUT_PATH)
        else:
            analyzer.create_comprehensive_plots()
        
        # Generate report
        summary_df = analyzer.generate_report()
        
        # Optional: Save summary to CSV
        summary_df.to_csv("imu_analysis_summary.csv", index=False)
        print(f"\nSummary saved to: imu_analysis_summary.csv")
        
    else: 
        print("Failed to load data. Please check the CSV file path.")


if __name__ == "__main__":

    main()

