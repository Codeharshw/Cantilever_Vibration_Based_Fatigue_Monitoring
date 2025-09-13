# Cantilever Fatigue Monitoring

A vibration-based fatigue detection system for metal cantilever structures using classical signal processing and deep learning approaches.

## Overview

This project monitors structural fatigue in cantilever beams through vibration analysis. An MPU9250 IMU sensor captures acceleration data from a motor-excited cantilever rod, processed via Arduino for real-time fatigue assessment.

**Key Features:**
- Classical modal analysis (frequency/damping shifts)
- Deep learning models (LSTM/GRU) for anomaly detection
- Real-time data acquisition with MPU9250 + Arduino
- Automated fatigue progression tracking

## Quick Start

```bash
git clone https://github.com/yourusername/cantilever-fatigue-monitoring.git
cd cantilever-fatigue-monitoring
pip install -r requirements.txt

# Run frequency analysis
python scripts/classical_analysis/frequency_analysis.py

# Run LSTM model
python scripts/deep_learning/lstm_fatigue_analysis.py
```

## Methods

### Classical Analysis
- **Frequency Tracking**: Detects stiffness reduction via natural frequency shifts
- **Damping Analysis**: Monitors crack-induced energy dissipation through peak broadening
- **RMS Analysis**: Tracks overall vibration amplitude changes

### Deep Learning
- **LSTM/GRU Models**: Learn baseline "healthy" vibration patterns
- **Anomaly Detection**: Identifies fatigue through prediction error increases
- **Time-Series Forecasting**: One-step-ahead prediction for real-time monitoring

## Project Structure

```
├── data/
│   └── calibrated_mpu9250_data.csv
├── scripts/
│   ├── classical_analysis/
│   │   ├── rms_analysis.py
│   │   ├── damping_analysis.py
│   │   └── frequency_analysis.py
│   └── deep_learning/
│       ├── lstm_fatigue_analysis.py
│       └── gru_fatigue_analysis.py
├── results/plots/
└── requirements.txt
```

## Requirements

- Python 3.8+
- NumPy, SciPy, Matplotlib
- TensorFlow/Keras (for deep learning)
- Arduino IDE (for data acquisition)

## Hardware Setup

- Cantilever beam (metal rod)
- MPU9250 IMU sensor
- Arduino microcontroller
- Motor for vibration excitation

## Usage

1. **Data Collection**: Upload Arduino sketch to collect vibration data
2. **Classical Analysis**: Run frequency/damping analysis scripts
3. **Deep Learning**: Train models on healthy data, then monitor for anomalies
4. **Results**: View generated plots in `results/plots/`

## Results

The system successfully detects fatigue progression through:
- Frequency reduction (stiffness loss)
- Damping increase (crack formation)
- LSTM prediction error spikes (behavioral changes)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-analysis`)
3. Commit changes (`git commit -am 'Add new analysis method'`)
4. Push to branch (`git push origin feature/new-analysis`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{cantilever_fatigue_monitoring,
  title={Cantilever Fatigue Monitoring},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/cantilever-fatigue-monitoring}
}
```

## Acknowledgments

Deep learning implementations adapted from "Time Series Forecasting using Deep Learning" by Ivan Gridin.
