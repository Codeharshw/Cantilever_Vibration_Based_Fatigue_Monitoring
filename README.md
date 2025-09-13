# Cantilever Vibration-Based Fatigue Monitoring

## Overview

This project presents a comprehensive framework for monitoring metal fatigue in a cantilever rod using vibration data collected from a real-world experimental setup. It leverages time-series analysis with both classical signal processing techniques and modern deep learning models (LSTM & GRU) to detect the subtle changes in a structure's dynamic response that indicate fatigue damage.

Our experimental setup involves a metal cantilever rod subjected to continuous vibration induced by a motor. An **MPU9250 IMU sensor** attached to the rod captures high-resolution acceleration data, which is then processed by an **Arduino microcontroller** and saved for analysis.

The core principle is that as a material fatigues—developing microscopic cracks and undergoing material degradation—its physical properties, specifically **stiffness** and **damping**, change. These changes manifest as measurable shifts in the vibration data. This repository offers the tools to capture and analyze these shifts, providing an early warning system for structural failure in an experimental context.

---

## Experimental Setup

The experiment is designed to induce and monitor fatigue in a cantilever metal rod under controlled conditions.

* **Cantilever Rod:** The metal specimen under test, securely clamped at one end to a wooden base, creating the cantilever configuration.
* **Vibration Source:** A small DC motor is directly coupled to the rod, inducing continuous, forced vibrations to accelerate the fatigue process.
* **Sensor:** An **MPU9250 Inertial Measurement Unit (IMU)** is securely taped to the cantilever rod, near its free end, to capture high-frequency acceleration data along the x, y, and z axes (`ax`, `ay`, `az`).
* **Data Acquisition:** An **Arduino Uno microcontroller** interfaces with the MPU9250 via I2C, reads the raw sensor data, and is typically configured to stream this data serially for logging into a `.csv` file (e.g., `calibrated_mpu9250_data.csv`).

---

## Methodologies Explored

This repository is organized into two primary analytical approaches, located in the `/scripts` directory:

### 1. Classical Signal Processing (`/scripts/classical_analysis`)

This physics-based method analyzes the raw vibration signal to extract direct indicators of structural health. It is fundamentally based on **Modal Analysis**, where changes in a structure's modal parameters (natural frequencies and damping ratios) directly correlate with physical damage and fatigue progression.

* **Resonance Frequency Analysis:** As the cantilever rod experiences fatigue and crack propagation, its effective stiffness decreases. This reduction in stiffness will cause its natural resonance frequency to exhibit a measurable **decrease**. This is a fundamental and highly sensitive indicator of structural damage.
* **Damping Analysis (Peak Width):** The formation and growth of internal cracks introduce new mechanisms for energy dissipation within the material (e.g., micro-friction between crack surfaces, plastic deformation at the crack tip). This phenomenon **increases** the system's overall damping, which is observed as a broadening (or widening) of the resonance peak in the frequency spectrum.

### 2. Time-Series Deep Learning (`/scripts/deep_learning`)

This data-driven method employs Recurrent Neural Networks (RNNs), specifically LSTMs and GRUs, to learn the complex temporal patterns characteristic of the "healthy" vibrating cantilever system. By rigorously training these models exclusively on vibration data obtained from the initial, undamaged phase of the experiment, they can then predict the expected vibration signal one step into the future.

* **Fatigue Detection:** As the cantilever rod experiences fatigue and its dynamic behavior changes due to crack initiation and propagation, its actual vibration signal will begin to deviate significantly from the patterns learned by the "healthy" model. This divergence between the actual and predicted values results in a **noticeable and sustained increase in the model's prediction error** over time, serving as a powerful and sensitive indicator of damage and fatigue progression.
* **Models Implemented:**
    * **LSTM (Long Short-Term Memory):** A robust RNN architecture particularly effective at capturing long-range dependencies and complex non-linear patterns in sequential data, making it suitable for modeling time-series vibration.
    * **GRU (Gated Recurrent Unit):** A more computationally efficient alternative to the LSTM, often delivering comparable performance in time-series prediction tasks with fewer parameters.

---

## Repository Structure

```
Cantilever_Vibration_Based_Fatigue_Monitoring/
├── data/
│   └── calibrated_mpu9250_data.csv
├── results/
│   └── plots/
├── scripts/
│   ├── classical_analysis/
│   │   ├── 01_rms_analysis.py
│   │   ├── 02_damping_analysis.py
│   │   └── 03_frequency_analysis.py
│   └── deep_learning/
│       ├── LSTM_fatigue_analysis.py
│       └── GRU_fatigue_analysis.py
├── .gitignore
├── README.md
├── requirements_classical.txt
└── requirements_dl.txt
```

---

## Getting Started

### Prerequisites

* Python 3.8+
* A Python virtual environment (highly recommended for dependency management)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[Your-Username]/Cantilever_Vibration_Based_Fatigue_Monitoring.git
    cd Cantilever_Vibration_Based_Fatigue_Monitoring
    ```

2.  **Set up your environment and install dependencies:**
    Choose the appropriate `requirements.txt` file based on the analysis you wish to run. It's recommended to set up separate virtual environments if you plan to run both classical and deep learning analyses frequently.

    * **For Classical Signal Processing:**
        ```bash
        # Activate your virtual environment first (e.g., source venv/bin/activate)
        pip install -r requirements_classical.txt
        ```
    * **For Deep Learning Analysis:**
        ```bash
        # Activate your virtual environment first
        pip install -r requirements_dl.txt
        ```

### Usage

All executable analysis scripts are located in the `/scripts` directory. Navigate to the appropriate subfolder and run the Python scripts directly from your terminal.

1.  **Running a Classical Analysis (e.g., Frequency Analysis):**
    ```bash
    cd scripts/classical_analysis
    python 03_frequency_analysis.py 
    ```

2.  **Running a Deep Learning Model (e.g., LSTM Analysis):**
    ```bash
    cd scripts/deep_learning
    python LSTM_fatigue_analysis.py 
    ```

Generated plots and results from the analyses will be automatically saved into the `/results/plots` directory.

---

## Acknowledgements

The deep learning models in this repository are adapted from the principles and code structures presented in the book **"Time Series Forecasting using Deep Learning" by Ivan Gridin**.
