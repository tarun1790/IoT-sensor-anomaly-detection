import os
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
import joblib

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

NAB_DATASETS = {
    "ambient_temperature": "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/ambient_temperature_system_failure.csv",
    "machine_temperature": "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/machine_temperature_system_failure.csv"
}

def download_nab_dataset(name):
    """Download a NAB dataset if not already present locally."""
    if name not in NAB_DATASETS:
        raise ValueError(f"Unknown NAB dataset name: {name}")
    
    local_path = os.path.join(DATA_DIR, f"{name}.csv")
    if os.path.exists(local_path):
        return local_path
        
    url = NAB_DATASETS[name]
    print(f"Downloading {name} dataset from {url}...")
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"Saved {name} dataset to {local_path}")
    except Exception as e:
        print(f"Failed to download {name}: {e}")
        # Fallback to generating a mock NAB temperature dataset if download fails
        generate_mock_nab_dataset(local_path, name)
        
    return local_path

def generate_mock_nab_dataset(path, name):
    """Generate a mock NAB dataset if download fails, preserving same format."""
    print(f"Generating mock {name} dataset...")
    np.random.seed(42)
    timestamps = pd.date_range(start="2026-06-01", periods=3000, freq="5min")
    
    if name == "ambient_temperature":
        # Ambient temperature has diurnal cycles and some sudden drops/spikes
        values = 70.0 + 10.0 * np.sin(2 * np.pi * np.arange(3000) / 288) + np.random.normal(0, 2.0, 3000)
        # Inject anomalies
        values[1200:1250] -= 15.0  # system failure / cooling drop
        values[2200:2220] += 12.0  # spike
    else:
        # Machine temperature: higher baseline, sudden spikes, gradual rise before failure
        values = 85.0 + 5.0 * np.sin(2 * np.pi * np.arange(3000) / 288) + np.random.normal(0, 1.5, 3000)
        # Inject gradual rise (fan failing)
        values[800:1000] += np.linspace(0, 25, 200)
        values[1000:1050] = 110.0 + np.random.normal(0, 1.0, 50)  # failure
        # Inject sudden anomaly
        values[2000:2020] += 20.0
        
    df = pd.DataFrame({"timestamp": timestamps, "value": values})
    df.to_csv(path, index=False)
    print(f"Generated mock data at {path}")

def generate_synthetic_industrial_data():
    """
    Generate a high-quality multi-sensor dataset.
    Simulates a factory machine with 4 sensors:
    - Temperature (C): diurnal cycle, fan failures (drift/spikes)
    - Pressure (PSI): steady state, leakage drops
    - Vibration (g): noise, bearing wear (variance/amplitude drift)
    - Power Consumption (kW): state-based jumps, electrical surges
    """
    path = os.path.join(DATA_DIR, "industrial_sensors.csv")
    if os.path.exists(path):
        return path
        
    print("Generating synthetic industrial multi-sensor dataset...")
    np.random.seed(42)
    n_records = 5000  # ~17.3 days of 5-min readings
    timestamps = pd.date_range(start="2026-06-01", periods=n_records, freq="5min")
    
    # 1. Temperature: 65C base + diurnal sine wave + noise
    t_base = 65.0
    t_cycle = 8.0 * np.sin(2 * np.pi * np.arange(n_records) / 288) # 288 steps per day
    t_noise = np.random.normal(0, 1.0, n_records)
    temperature = t_base + t_cycle + t_noise
    
    # 2. Pressure: 45 PSI base + minor cycles + noise
    p_base = 45.0
    p_cycle = 2.0 * np.cos(2 * np.pi * np.arange(n_records) / 144)
    p_noise = np.random.normal(0, 0.8, n_records)
    pressure = p_base + p_cycle + p_noise
    
    # 3. Vibration: 2.2g base + high frequency noise
    v_base = 2.2
    v_noise = np.random.normal(0, 0.25, n_records)
    vibration = v_base + v_noise
    
    # 4. Power Consumption: 15 kW base + operational step changes (machine speed) + noise
    power_base = 15.0
    # Simulate machine operation cycles (active vs idle / low-power states)
    op_states = np.zeros(n_records)
    current_state = 0
    state_durations = [144, 72, 288, 72] # states in steps
    state_values = [0, 5.0, 10.0, -3.0] # deviation from base
    
    idx = 0
    while idx < n_records:
        dur = np.random.choice(state_durations)
        val = np.random.choice(state_values)
        end = min(idx + dur, n_records)
        op_states[idx:end] = val
        idx = end
    power = power_base + op_states + np.random.normal(0, 0.5, n_records)
    
    # Labels for validation
    anomaly_labels = np.zeros(n_records, dtype=int)
    
    # --- Inject Anomalies ---
    
    # Temperature: Slow Drift / Overheating anomaly (Cooling Fan degradation)
    # Timestep 1500 to 1800 (approx 25 hours)
    temperature[1500:1800] += np.linspace(0, 18, 300)
    anomaly_labels[1600:1800] = 1 # label after it exceeds normal bounds
    
    # Pressure: Sudden Pressure Drop / Gas leak
    # Timestep 2800 to 2860 (5 hours)
    pressure[2800:2860] -= 15.0
    anomaly_labels[2800:2860] = 1
    
    # Vibration: Bearing degradation (gradual rise in mean and variance)
    # Timestep 4200 onwards
    vibration[4200:] += np.linspace(0, 2.5, n_records - 4200)
    vibration[4200:] += np.random.normal(0, np.linspace(0.1, 0.8, n_records - 4200), n_records - 4200)
    anomaly_labels[4400:] = 1
    
    # Power: Extreme current surges / Spike
    # Let's inject a few random spikes
    spike_indices = [500, 1200, 2200, 3100, 3700]
    for sp_idx in spike_indices:
        power[sp_idx] += 18.0
        anomaly_labels[sp_idx] = 1
        # Also heat spikes slightly
        temperature[sp_idx] += 3.0
        
    df = pd.DataFrame({
        "timestamp": timestamps,
        "temperature": temperature,
        "pressure": pressure,
        "vibration": vibration,
        "power": power,
        "anomaly": anomaly_labels
    })
    
    df.to_csv(path, index=False)
    print(f"Saved synthetic industrial dataset to {path}")
    return path

class DatasetLoader:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.raw_path = None
        self.df = None
        self.scaler = MinMaxScaler()
        self.feature_cols = []
        self.load_data()
        
    def load_data(self):
        if self.dataset_name == "industrial":
            self.raw_path = generate_synthetic_industrial_data()
            self.df = pd.read_csv(self.raw_path)
            self.feature_cols = ["temperature", "pressure", "vibration", "power"]
        elif self.dataset_name in ["ambient_temperature", "machine_temperature"]:
            self.raw_path = download_nab_dataset(self.dataset_name)
            self.df = pd.read_csv(self.raw_path)
            self.feature_cols = ["value"]
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
            
        # Parse timestamps
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
        
    def get_train_val_data(self, train_split=0.8):
        """Preprocesses and returns train/validation splits scaled."""
        # Fit scaler on features
        scaled_features = self.scaler.fit_transform(self.df[self.feature_cols])
        
        # Save scaler for online streaming scaling
        scaler_path = os.path.join(DATA_DIR, f"{self.dataset_name}_scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        
        n_train = int(len(scaled_features) * train_split)
        train_data = scaled_features[:n_train]
        val_data = scaled_features[n_train:]
        
        return train_data, val_data, self.scaler

    def create_windows(self, data, window_size=12):
        """
        Creates rolling windows for time-series forecasting (LSTM) or sequence reconstruction (Autoencoder).
        For forecasting: X is shape (n, window_size, features), y is (n, features).
        """
        X = []
        y = []
        for i in range(len(data) - window_size):
            X.append(data[i : i + window_size])
            y.append(data[i + window_size])
        return np.array(X), np.array(y)

if __name__ == "__main__":
    # Test generation/downloads
    print("Testing data manager...")
    generate_synthetic_industrial_data()
    download_nab_dataset("ambient_temperature")
    download_nab_dataset("machine_temperature")
    
    loader = DatasetLoader("industrial")
    train, val, scaler = loader.get_train_val_data()
    X_train, y_train = loader.create_windows(train, window_size=12)
    print(f"Industrial train windows shape: {X_train.shape}, labels shape: {y_train.shape}")
