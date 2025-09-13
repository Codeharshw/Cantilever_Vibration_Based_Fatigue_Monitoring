# Cantilever Vibration-Based Fatigue Monitoring

## Overview

This project provides a comprehensive framework for monitoring metal fatigue in a cantilever rod using vibration data. It leverages time-series analysis with both classical signal processing techniques and modern deep learning models (LSTM & GRU) to detect the subtle changes in a structure's dynamic response that indicate fatigue damage.

The core principle is that as a material fatigues, its physical properties—specifically **stiffness** and **damping**—change. These changes manifest as measurable shifts in the vibration data. This repository offers the tools to capture and analyze these shifts, providing an early warning system for structural failure.

![Vibration Analysis Concept](https://i.imgur.com/rS2B911.png)
*(Conceptual diagram showing how vibration data is used for health monitoring)*

---

## Methodologies Explored

This repository is organized into two primary analytical approaches, located in the `/scripts` directory:

### 1. Classical Signal Processing (`/scripts/classical_analysis`)

This physics-based method analyzes the vibration signal to extract direct indicators of structural health. It is based on **Modal Analysis**, where changes in the modal parameters (frequency and damping) correlate directly with physical damage.

* **Resonance Frequency Analysis:** Fatigue cracks reduce the rod's stiffness, causing a **decrease** in its natural resonance frequency.
* **Damping Analysis (Peak Width):** Cracks introduce energy dissipation mechanisms (e.g., friction), which **increases** the system's damping. This is observed as a widening of the resonance peak in the frequency spectrum.

### 2. Time-Series Deep Learning (`/scripts/deep_learning`)

This data-driven method uses Recurrent Neural Networks (RNNs) to learn the complex temporal patterns of a "healthy" vibrating system. By training on the initial, undamaged phase of the experiment, the models can predict the expected vibration signal one step into the future.

* **Fatigue Detection:** As fatigue develops, the rod's behavior diverges from the learned "healthy" patterns. This causes the model's **prediction error to increase significantly**, serving as a powerful and sensitive indicator of damage.
* **Models Implemented:**
    * **LSTM (Long Short-Term Memory):** A sophisticated RNN architecture capable of learning long-range dependencies in the data.
    * **GRU (Gated Recurrent Unit):** A more streamlined version of the LSTM, often delivering similar performance with greater computational efficiency.

---

## Repository Structure

```
Cantilever_Vibration_Based_Fatigue_Monitoring/
├── data/
│   └── calibrated_mpu9_data.csv
├── results/
│   └── plots/
├── scripts/
│   ├── classical_analysis/
│   │   ├── 01_rms_analysis.py
│   │   ├── ...
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
* A Python virtual environment (recommended)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[Your-Username]/Cantilever_Vibration_Based_Fatigue_Monitoring.git
    cd Cantilever_Vibration_Based_Fatigue_Monitoring
    ```

2.  **Set up your environment and install dependencies:**
    Choose the appropriate requirements file based on the analysis you wish to run.

    * **For Classical Signal Processing:**
        ```bash
        pip install -r requirements_classical.txt
        ```
    * **For Deep Learning Analysis:**
        ```bash
        pip install -r requirements_dl.txt
        ```

### Usage

All executable scripts are located in the `/scripts` directory. Navigate to the appropriate subfolder and run the Python scripts directly.

1.  **Running a Classical Analysis:**
    ```bash
    cd scripts/classical_analysis
    python 01_frequency_analysis.py
    ```

2.  **Running a Deep Learning Model:**
    ```bash
    cd scripts/deep_learning
    python LSTM_fatigue_analysis.py
    ```

Generated plots and results will be saved in the `/results/plots` directory.

---

## Example Result: Fatigue Detection using LSTM

The plot below shows a key result from the LSTM model. The model, trained only on "healthy" data, shows a low prediction error initially. As fatigue damage accumulates in the test set, the actual vibration deviates from the model's predictions, causing a noticeable and sustained increase in the error—a clear sign of structural change.



This rising error trend is the primary indicator of fatigue progression when using the deep learning approach.

---

## Acknowledgements

The deep learning models in this repository are adapted from the principles and code structures presented in the book **"Time Series Forecasting using Deep Learning" by Ivan Gridin**.
