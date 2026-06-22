# Sensor Anomaly Detection Dashboard

### Internship Details
* **Intern Name**: Jampani Tarun Sai
* **Intern ID**: CITS1344
* **Internship Duration**: 12 Weeks

---

## Project Overview

An end-to-end Industrial IoT sensor health, accuracy, and anomaly detection monitoring pipeline. This system features a real-time stateful telemetry simulator, background GPU-accelerated machine learning training and inference, and a clean minimalist white, black, and crimson red dashboard.

## Key Features

1. **Multi-Model Anomaly Detection Pipeline**:
   - **PyTorch Autoencoder (GPU/CUDA)**: Performs sequence reconstruction. High reconstruction errors are mapped directly to **Sensor Accuracy** metrics.
   - **PyTorch LSTM Forecaster (GPU/CUDA)**: Forecasts next-step parameters to detect contextual and temporal anomalies.
   - **scikit-learn Isolation Forest**: Provides unsupervised point anomaly detection.

2. **Real-time Stateful Telemetry Simulator**:
   - Streams multi-sensor signals (Temperature, Pressure, Vibration, Power Consumption) with diurnal operational cycles.
   - Supports live fault injection to verify model responses: *Temperature Spikes, Pressure Leaks, Bearing Wear, Calibration Drifts, Sensor Noise, and Output Flatlines*.

3. **Dynamic Sensor Status Indicators**:
   - **Sensor Health**: Calculates physical sensor structural health based on signal variance, baseline drift, and rolling anomalies.
   - **Sensor Accuracy**: Represents sensor calibration/reading accuracy computed using Autoencoder reconstruction loss.

4. **Actionable AI Event Log**:
   - Flags anomalies in real time and provides context-aware maintenance recommendations (e.g., check pipe seals, relubricate bearing, recalibrate transmitters).

5. **Clean Minimalist Dashboard**:
   - High-contrast white background with black typography and crimson red accents.
   - Smooth animated card-lift scaling and shadow glows on hover.

---

## Directory Structure

```
Anomaly Detection in IoT Sensors/
├── requirements.txt (Python Dependencies)
├── main.py (FastAPI Backend Server & REST Endpoints)
├── data_manager.py (Mock data generator & NAB dataset downloader)
├── simulator.py (Stateful simulator with fault injector)
├── README.md (Project Documentation)
├── models/
│   ├── __init__.py
│   ├── autoencoder_model.py (PyTorch Autoencoder module)
│   ├── lstm_model.py (PyTorch LSTM forecaster module)
│   ├── isolation_forest_model.py (scikit-learn Isolation Forest wrapper)
│   ├── trainer.py (Background training coordinator)
│   └── saved_models/ (Trained weights/models)
└── static/
    ├── index.html (Dashboard layout)
    ├── styles.css (Clean light theme styles & animations)
    └── app.js (Chart.js plotting & UI updates)
```

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for PyTorch acceleration)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/tarun1790/IoT-sensor-anomaly-detection.git
   cd IoT-sensor-anomaly-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the FastAPI server:
   ```bash
   python main.py
   ```

4. Open your browser and navigate to:
   ```
   http://127.0.0.1:8000/
   ```
