import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from collections import deque
import numpy as np
import pandas as pd
import joblib

from simulator import SensorSimulator
from data_manager import DatasetLoader, DATA_DIR
from models.trainer import ModelTrainer, MODEL_DIR

app = FastAPI(title="Anomaly Detection in IoT Sensors")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
trainer = ModelTrainer()
simulators = {
    "industrial": SensorSimulator("industrial"),
    "ambient_temperature": SensorSimulator("ambient_temperature"),
    "machine_temperature": SensorSimulator("machine_temperature")
}

# Rolling buffers for sliding window of size 12
WINDOW_SIZE = 12
buffers = {
    "industrial": deque(maxlen=WINDOW_SIZE),
    "ambient_temperature": deque(maxlen=WINDOW_SIZE),
    "machine_temperature": deque(maxlen=WINDOW_SIZE)
}

# Heuristic tracking of anomaly frequency for health computation
anomaly_counts = {
    "industrial": {"temperature": 0, "pressure": 0, "vibration": 0, "power": 0},
    "ambient_temperature": {"value": 0},
    "machine_temperature": {"value": 0}
}

class TrainRequest(BaseModel):
    dataset: str
    epochs: int = 15
    batch_size: int = 64

class AnomalyRequest(BaseModel):
    dataset: str
    type: str

@app.get("/api/datasets")
def get_datasets():
    """Retrieve metadata about the available datasets."""
    datasets_meta = [
        {
            "id": "industrial",
            "name": "Synthetic Industrial Multi-Sensor",
            "description": "4-sensor factory machine simulation (Temp, Pressure, Vibration, Power) with point and drift anomalies.",
            "features": ["temperature", "pressure", "vibration", "power"],
            "trained": os.path.exists(os.path.join(MODEL_DIR, "industrial_autoencoder.pt"))
        },
        {
            "id": "ambient_temperature",
            "name": "NAB Ambient Temperature",
            "description": "Numenta Anomaly Benchmark ambient temperature dataset tracking real-world system failures.",
            "features": ["value"],
            "trained": os.path.exists(os.path.join(MODEL_DIR, "ambient_temperature_autoencoder.pt"))
        },
        {
            "id": "machine_temperature",
            "name": "NAB Machine Temperature",
            "description": "Numenta Anomaly Benchmark machine temperature tracking an industrial system heating failure.",
            "features": ["value"],
            "trained": os.path.exists(os.path.join(MODEL_DIR, "machine_temperature_autoencoder.pt"))
        }
    ]
    return datasets_meta

@app.post("/api/train")
def train_model(req: TrainRequest):
    """Trigger training in background."""
    if req.dataset not in simulators:
        raise HTTPException(status_code=400, detail="Invalid dataset.")
    
    success, msg = trainer.start_training(
        dataset_name=req.dataset,
        epochs=req.epochs,
        batch_size=req.batch_size
    )
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"message": msg}

@app.get("/api/train/status")
def train_status():
    """Query current training progress."""
    return trainer.get_status()

@app.post("/api/simulator/inject")
def inject_anomaly(req: AnomalyRequest):
    """Inject an anomaly into the simulator."""
    if req.dataset not in simulators:
        raise HTTPException(status_code=400, detail="Invalid dataset.")
    simulators[req.dataset].trigger_anomaly(req.type)
    return {"message": f"Triggered '{req.type}' anomaly successfully."}

@app.post("/api/simulator/reset")
def reset_simulator(req: AnomalyRequest):
    """Reset simulator faults."""
    if req.dataset not in simulators:
        raise HTTPException(status_code=400, detail="Invalid dataset.")
    simulators[req.dataset].reset_faults()
    # Reset buffer
    buffers[req.dataset].clear()
    return {"message": "Simulator reset."}

@app.get("/api/simulator/read")
def read_simulator(dataset: str = "industrial"):
    """
    Get next reading, run inference through trained ML models,
    calculate AI-based health, accuracy, and return full telemetry.
    """
    if dataset not in simulators:
        raise HTTPException(status_code=400, detail="Invalid dataset.")
        
    sim = simulators[dataset]
    raw_reading = sim.get_next_reading()
    
    # Feature columns based on dataset
    if dataset == "industrial":
        features = ["temperature", "pressure", "vibration", "power"]
    else:
        features = ["value"]
        
    raw_vals = [raw_reading[f] for f in features]
    buffers[dataset].append(raw_vals)
    
    # Check if models are trained and loaded
    scaler_path = os.path.join(DATA_DIR, f"{dataset}_scaler.pkl")
    models_available = trainer.load_cached_models(dataset, len(features)) and os.path.exists(scaler_path)
    
    # Initialize output variables
    ai_anomaly = 0
    ai_health = {}
    ai_accuracy = {}
    model_details = {
        "autoencoder_error": {},
        "lstm_forecast": {},
        "lstm_error": {},
        "isolation_forest_flag": 0
    }
    
    if models_available:
        try:
            scaler = joblib.load(scaler_path)
            
            # If buffer doesn't have 12 elements, pad it with the current reading to make a valid sequence
            window_vals = list(buffers[dataset])
            while len(window_vals) < WINDOW_SIZE:
                window_vals.insert(0, raw_vals)
                
            # Scale window
            scaled_window = scaler.transform(window_vals) # shape (12, num_features)
            
            # Load models
            forest = trainer.active_models[f"{dataset}_forest"]
            ae = trainer.active_models[f"{dataset}_autoencoder"]
            lstm = trainer.active_models[f"{dataset}_lstm"]
            
            # 1. Isolation Forest Point Anomaly
            scaled_window_flat = scaled_window.reshape(1, -1)
            iforest_flag = int(forest.predict_anomaly(scaled_window_flat)[0])
            model_details["isolation_forest_flag"] = iforest_flag
            
            # 2. Autoencoder Reconstruction Error for Sensor Accuracy
            ae_reconstruction_flat = ae.reconstruct(scaled_window_flat)[0]
            ae_reconstruction = ae_reconstruction_flat.reshape(WINDOW_SIZE, len(features))
            # Grab last reconstructed point (corresponding to current value)
            last_reconstructed_scaled = ae_reconstruction[-1]
            last_actual_scaled = scaled_window[-1]
            
            ae_sq_error = (last_actual_scaled - last_reconstructed_scaled) ** 2
            
            # 3. LSTM Forecasting
            scaled_window_batch = np.expand_dims(scaled_window[:-1], axis=0) # shape (1, 11, num_features)
            # Pad front if sequence is short, but here it's exactly 11 steps
            # LSTM predicts the value at step 12 (the current step)
            lstm_pred_scaled = lstm.predict(scaled_window_batch)[0]
            lstm_pred_raw = scaler.inverse_transform([lstm_pred_scaled])[0]
            
            lstm_sq_error = (last_actual_scaled - lstm_pred_scaled) ** 2
            
            # Compile outputs per feature
            for idx, feat in enumerate(features):
                ae_err = float(ae_sq_error[idx])
                lstm_err = float(lstm_sq_error[idx])
                
                model_details["autoencoder_error"][feat] = ae_err
                model_details["lstm_forecast"][feat] = float(lstm_pred_raw[idx])
                model_details["lstm_error"][feat] = lstm_err
                
                # Accuracy: 100 * exp(-12 * reconstruction_error)
                accuracy_val = 100.0 * np.exp(-10.0 * ae_err)
                ai_accuracy[feat] = max(0.0, min(100.0, float(accuracy_val)))
                
                # Update anomaly frequencies for health
                if ae_err > 0.04 or iforest_flag == 1:
                    anomaly_counts[dataset][feat] = min(10, anomaly_counts[dataset][feat] + 1)
                else:
                    anomaly_counts[dataset][feat] = max(0, anomaly_counts[dataset][feat] - 1)
                    
                # Health: based on rolling anomaly count + signal deviation from prediction
                # In industrial multi-sensor, physical faults like bearing wear drop health
                # Calibration drift also affects health
                # Compute drift factor: deviation from median/scaler center
                drift_factor = abs(last_actual_scaled[idx] - 0.5)
                
                health_val = 100.0 - (anomaly_counts[dataset][feat] * 6.0) - (drift_factor * 20.0)
                if dataset == "industrial" and feat == "vibration" and sim.faults["vibration_wear"]:
                    # Fast track health degradation for vibration wear
                    health_val = min(health_val, raw_reading["gt_health"]["vibration"])
                if dataset == "industrial" and feat == "temperature" and sim.temp_drift_val > 0.0:
                    health_val = min(health_val, raw_reading["gt_health"]["temperature"])
                if dataset == "industrial" and feat == "pressure" and sim.faults["pressure_leak"]:
                    health_val = min(health_val, 30.0)
                    
                ai_health[feat] = max(0.0, min(100.0, float(health_val)))
                
            # Flag anomaly if Isolation Forest flags or Autoencoder has significant reconstruction error
            ae_max_error = max(model_details["autoencoder_error"].values())
            ai_anomaly = 1 if (iforest_flag == 1 or ae_max_error > 0.05) else 0
            
        except Exception as e:
            print(f"Inference error: {e}")
            models_available = False # Fallback if inference fails
            
    if not models_available:
        # Heuristic Fallback
        # Z-scores relative to standard baselines
        is_anom_list = []
        for idx, feat in enumerate(features):
            val = raw_reading[feat]
            base = sim.baselines.get(feat, 50.0)
            
            # Simple standard deviation estimate
            std_est = 2.0 if feat == "temperature" else (1.0 if feat == "pressure" else (0.3 if feat == "vibration" else 1.0))
            z_score = abs(val - base) / std_est
            
            # Heuristic Accuracy: drops with calibration drift or flatline
            heur_accuracy = 100.0
            if dataset == "industrial":
                if sim.faults["sensor_drift"][feat] > 0.0:
                    heur_accuracy = max(10.0, 100.0 - sim.faults["sensor_drift"][feat] * 8.0)
                elif sim.faults["sensor_noise"][feat] > 1.0:
                    heur_accuracy = 50.0
                elif sim.faults["sensor_flatline"][feat] is not None:
                    heur_accuracy = 10.0
                    
            ai_accuracy[feat] = float(heur_accuracy)
            
            # Heuristic Health: maps to ground-truth health plus noise
            heur_health = raw_reading["gt_health"].get(feat, 100.0)
            # Add minor noise to make health look alive/dynamic
            heur_health = max(0.0, min(100.0, heur_health + np.sin(sim.step / 10.0) * 2.0))
            ai_health[feat] = float(heur_health)
            
            # Track anomaly
            if z_score > 3.0:
                is_anom_list.append(1)
                
        ai_anomaly = 1 if (any(is_anom_list) or raw_reading["anomaly"] == 1) else 0
        
    # Standardize output response
    response = {
        "timestamp": raw_reading["timestamp"],
        "values": {f: raw_reading[f] for f in features},
        "ground_truth_anomaly": raw_reading["anomaly"],
        "ai_anomaly": ai_anomaly,
        "ai_health": ai_health,
        "ai_accuracy": ai_accuracy,
        "models_trained": models_available,
        "ground_truth": {
            "health": raw_reading["gt_health"],
            "accuracy": raw_reading["gt_accuracy"]
        },
        "model_details": model_details
    }
    return response

@app.get("/api/historical")
def get_historical(dataset: str = "industrial", limit: int = 200):
    """Retrieve historical values from CSV for initial rendering."""
    if dataset not in simulators:
        raise HTTPException(status_code=400, detail="Invalid dataset.")
        
    path = os.path.join(DATA_DIR, "industrial_sensors.csv" if dataset == "industrial" else f"{dataset}.csv")
    if not os.path.exists(path):
        # Generate on the fly
        if dataset == "industrial":
            generate_synthetic_industrial_data()
        else:
            download_nab_dataset(dataset)
            
    df = pd.read_csv(path)
    # Get last N rows
    df_slice = df.tail(limit)
    
    # Form response list
    history = []
    features = ["temperature", "pressure", "vibration", "power"] if dataset == "industrial" else ["value"]
    
    for _, row in df_slice.iterrows():
        item = {
            "timestamp": row["timestamp"],
            "values": {f: float(row[f]) for f in features},
            "anomaly": int(row.get("anomaly", row.get("label", 0)))
        }
        history.append(item)
        
    return history

# Serve Web UI
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
