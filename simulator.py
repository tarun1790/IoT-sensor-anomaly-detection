import time
import numpy as np
import pandas as pd

class SensorSimulator:
    def __init__(self, dataset_name="industrial"):
        self.dataset_name = dataset_name
        self.step = 0
        self.np_random = np.random.RandomState(int(time.time()) % 100000)
        
        # Injectable faults state
        self.faults = {
            "temp_spike": False,       # Sudden spike
            "temp_drift": False,       # Slow upward drift
            "pressure_leak": False,    # Sudden drop in pressure
            "vibration_wear": False,   # Gradual increase in vibration amplitude/variance
            "sensor_drift": {          # Specific sensor calibration drift
                "temperature": 0.0,
                "pressure": 0.0,
                "vibration": 0.0,
                "power": 0.0
            },
            "sensor_noise": {          # Specific sensor noise multiplier
                "temperature": 1.0,
                "pressure": 1.0,
                "vibration": 1.0,
                "power": 1.0
            },
            "sensor_flatline": {       # Sensor completely freezes
                "temperature": None,
                "pressure": None,
                "vibration": None,
                "power": None
            }
        }
        
        # Physical stats/baselines
        self.baselines = {
            "temperature": 65.0,
            "pressure": 45.0,
            "vibration": 2.2,
            "power": 15.0
        }
        
        # Keep track of active wear values
        self.vibration_degradation = 0.0
        self.temp_drift_val = 0.0

    def trigger_anomaly(self, anomaly_type):
        """Enable specific anomalies in the simulator."""
        if anomaly_type == "temp_spike":
            self.faults["temp_spike"] = True
        elif anomaly_type == "temp_drift":
            self.faults["temp_drift"] = True
        elif anomaly_type == "pressure_leak":
            self.faults["pressure_leak"] = True
        elif anomaly_type == "vibration_wear":
            self.faults["vibration_wear"] = True
        elif anomaly_type == "sensor_drift":
            # Apply drift to a random sensor (e.g. temperature)
            self.faults["sensor_drift"]["temperature"] += 1.0 # Increments each step
        elif anomaly_type == "sensor_noise":
            # Apply high noise to pressure sensor
            self.faults["sensor_noise"]["pressure"] = 5.0
        elif anomaly_type == "sensor_flatline":
            # Freeze vibration sensor at its last value
            self.faults["sensor_flatline"]["vibration"] = 2.2
        elif anomaly_type == "reset":
            self.reset_faults()
            
    def reset_faults(self):
        self.faults["temp_spike"] = False
        self.faults["temp_drift"] = False
        self.faults["pressure_leak"] = False
        self.faults["vibration_wear"] = False
        self.vibration_degradation = 0.0
        self.temp_drift_val = 0.0
        for k in self.faults["sensor_drift"]:
            self.faults["sensor_drift"][k] = 0.0
        for k in self.faults["sensor_noise"]:
            self.faults["sensor_noise"][k] = 1.0
        for k in self.faults["sensor_flatline"]:
            self.faults["sensor_flatline"][k] = None

    def get_next_reading(self):
        """Generates the next sensor reading dictionary based on simulation step."""
        self.step += 1
        
        if self.dataset_name == "industrial":
            # Time components
            # 288 steps of 5-min intervals in a day
            diurnal_cycle = np.sin(2 * np.pi * self.step / 288)
            
            # --- Baseline Signals ---
            # Temperature
            temp = self.baselines["temperature"] + 8.0 * diurnal_cycle + self.np_random.normal(0, 1.0)
            # Pressure
            pres = self.baselines["pressure"] + 2.0 * np.cos(2 * np.pi * self.step / 144) + self.np_random.normal(0, 0.8)
            # Vibration
            vib = self.baselines["vibration"] + self.np_random.normal(0, 0.25)
            # Power
            power = self.baselines["power"] + self.np_random.normal(0, 0.5)
            
            # Operational cycles (periodic high load)
            op_load = 5.0 if (self.step % 288) < 144 else 0.0
            power += op_load
            
            # --- Apply Faults / Anomalies ---
            is_anomaly = 0
            
            # 1. Temperature Spike (Sudden overheating)
            if self.faults["temp_spike"]:
                temp += 25.0
                is_anomaly = 1
                self.faults["temp_spike"] = False # Reset spike (one-off)
                
            # 2. Temperature Drift (Cooling breakdown)
            if self.faults["temp_drift"]:
                self.temp_drift_val += 0.2
                temp += self.temp_drift_val
                is_anomaly = 1
                # Cap drift
                if self.temp_drift_val > 25.0:
                    self.faults["temp_drift"] = False # Reset drift trigger but keep value or just stabilize
            
            # 3. Pressure Leak (Sudden drop)
            if self.faults["pressure_leak"]:
                pres -= 18.0
                is_anomaly = 1
                
            # 4. Vibration Degradation (Bearing wear)
            if self.faults["vibration_wear"]:
                self.vibration_degradation += 0.02
                vib += self.vibration_degradation + self.np_random.normal(0, self.vibration_degradation * 0.3)
                is_anomaly = 1
                
            # --- Apply Sensor-Specific Failures (Drift, Noise, Flatline) ---
            # Sensor Calibration Drift (linear deviation)
            temp += self.faults["sensor_drift"]["temperature"]
            pres += self.faults["sensor_drift"]["pressure"]
            vib += self.faults["sensor_drift"]["vibration"]
            power += self.faults["sensor_drift"]["power"]
            if any(val != 0.0 for val in self.faults["sensor_drift"].values()):
                is_anomaly = 1
                # Increment active drifts
                for k in self.faults["sensor_drift"]:
                    if self.faults["sensor_drift"][k] != 0.0:
                        self.faults["sensor_drift"][k] += 0.1
                        
            # Sensor Noise (amplified local variance)
            temp = temp + self.np_random.normal(0, 1.0) * (self.faults["sensor_noise"]["temperature"] - 1.0)
            pres = pres + self.np_random.normal(0, 0.8) * (self.faults["sensor_noise"]["pressure"] - 1.0)
            vib = vib + self.np_random.normal(0, 0.25) * (self.faults["sensor_noise"]["vibration"] - 1.0)
            power = power + self.np_random.normal(0, 0.5) * (self.faults["sensor_noise"]["power"] - 1.0)
            if any(val > 1.0 for val in self.faults["sensor_noise"].values()):
                is_anomaly = 1
                
            # Sensor Flatline (freezes output)
            if self.faults["sensor_flatline"]["temperature"] is not None:
                temp = self.faults["sensor_flatline"]["temperature"]
                is_anomaly = 1
            if self.faults["sensor_flatline"]["pressure"] is not None:
                pres = self.faults["sensor_flatline"]["pressure"]
                is_anomaly = 1
            if self.faults["sensor_flatline"]["vibration"] is not None:
                vib = self.faults["sensor_flatline"]["vibration"]
                is_anomaly = 1
            if self.faults["sensor_flatline"]["power"] is not None:
                power = self.faults["sensor_flatline"]["power"]
                is_anomaly = 1
                
            # Compute Ground Truth Health/Accuracy for each sensor
            # Health reflects the physical state of the component (e.g. pressure leak, bearing wear decrease health)
            # Accuracy reflects the reliability of the reading (e.g. sensor drift, noise, flatline decrease accuracy)
            
            healths = {
                "temperature": max(0.0, 100.0 - self.temp_drift_val * 3.0),
                "pressure": 30.0 if self.faults["pressure_leak"] else 100.0,
                "vibration": max(0.0, 100.0 - self.vibration_degradation * 15.0),
                "power": 100.0
            }
            
            accuracies = {
                "temperature": max(0.0, 100.0 - self.faults["sensor_drift"]["temperature"] * 10.0),
                "pressure": 50.0 if self.faults["sensor_noise"]["pressure"] > 1.0 else 100.0,
                "vibration": 10.0 if self.faults["sensor_flatline"]["vibration"] is not None else 100.0,
                "power": 100.0
            }
            
            reading = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "temperature": float(temp),
                "pressure": float(pres),
                "vibration": float(vib),
                "power": float(power),
                "anomaly": int(is_anomaly),
                "gt_health": healths,
                "gt_accuracy": accuracies
            }
            
        else: # Single value dataset (ambient_temperature / machine_temperature)
            # Fetch a baseline cycle
            diurnal_cycle = np.sin(2 * np.pi * self.step / 288)
            base = 70.0 if self.dataset_name == "ambient_temperature" else 85.0
            val = base + 5.0 * diurnal_cycle + self.np_random.normal(0, 1.5)
            
            is_anomaly = 0
            # Inject generic spike
            if self.faults["temp_spike"]:
                val += 20.0
                is_anomaly = 1
                self.faults["temp_spike"] = False
            
            # Inject drift
            if self.faults["temp_drift"]:
                self.temp_drift_val += 0.25
                val += self.temp_drift_val
                is_anomaly = 1
                if self.temp_drift_val > 25.0:
                    self.faults["temp_drift"] = False
                    
            reading = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "value": float(val),
                "anomaly": int(is_anomaly),
                "gt_health": {"value": max(0.0, 100.0 - self.temp_drift_val * 4.0)},
                "gt_accuracy": {"value": 100.0}
            }
            
        return reading

if __name__ == "__main__":
    print("Testing simulator...")
    sim = SensorSimulator("industrial")
    for _ in range(5):
        print(sim.get_next_reading())
    
    print("\nTriggering pressure leak...")
    sim.trigger_anomaly("pressure_leak")
    print(sim.get_next_reading())
