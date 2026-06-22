import os
import time
import threading
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from data_manager import DatasetLoader
from models.autoencoder_model import Autoencoder
from models.lstm_model import LSTMForecaster
from models.isolation_forest_model import IsolationForestWrapper

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

class ModelTrainer:
    def __init__(self):
        self.status = {
            "is_training": False,
            "dataset_name": "",
            "current_epoch": 0,
            "total_epochs": 0,
            "train_loss": [],
            "val_loss": [],
            "device_used": "cpu",
            "message": "Idle",
            "training_speed": "",
            "eta_seconds": 0
        }
        self.lock = threading.Lock()
        self.active_models = {}

    def get_status(self):
        with self.lock:
            return self.status.copy()

    def update_status(self, **kwargs):
        with self.lock:
            for k, v in kwargs.items():
                if k in self.status:
                    self.status[k] = v

    def start_training(self, dataset_name, epochs=20, batch_size=64, learning_rate=0.001):
        """Launches training in a background thread."""
        with self.lock:
            if self.status["is_training"]:
                return False, "Training is already in progress."
            
            self.status["is_training"] = True
            self.status["dataset_name"] = dataset_name
            self.status["current_epoch"] = 0
            self.status["total_epochs"] = epochs
            self.status["train_loss"] = []
            self.status["val_loss"] = []
            self.status["message"] = "Loading data..."
            self.status["device_used"] = "cuda" if torch.cuda.is_available() else "cpu"
            self.status["training_speed"] = ""
            self.status["eta_seconds"] = 0

        thread = threading.Thread(
            target=self._train_job,
            args=(dataset_name, epochs, batch_size, learning_rate),
            daemon=True
        )
        thread.start()
        return True, "Training started."

    def _train_job(self, dataset_name, epochs, batch_size, learning_rate):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.update_status(message="Preprocessing data...")
            
            # Load and preprocess dataset
            loader = DatasetLoader(dataset_name)
            train_data, val_data, scaler = loader.get_train_val_data()
            
            # Window parameters
            window_size = 12
            X_train, y_train = loader.create_windows(train_data, window_size)
            X_val, y_val = loader.create_windows(val_data, window_size)
            
            n_features = len(loader.feature_cols)
            
            # 1. Train Isolation Forest
            self.update_status(message="Training Isolation Forest...")
            # For Isolation Forest, we train on the last values of the training sequences (flattened or just points)
            # Flattened window for point/sequence isolation forest:
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            
            if Forest := self.active_models.get(f"{dataset_name}_forest"):
                if hasattr(Forest, 'model'):
                    del Forest
            
            forest_model = IsolationForestWrapper()
            forest_model.fit(X_train_flat)
            
            # Save forest model
            forest_path = os.path.join(MODEL_DIR, f"{dataset_name}_forest.pkl")
            forest_model.save(forest_path)
            
            # 2. Train PyTorch Autoencoder
            self.update_status(message="Training Autoencoder (PyTorch on GPU)...")
            autoencoder = Autoencoder(input_dim=window_size * n_features)
            ae_optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
            ae_criterion = nn.MSELoss()
            
            # Prepare loaders
            # Autoencoder reconstructs the full window, so X is target
            train_ae_dataset = TensorDataset(
                torch.tensor(X_train_flat, dtype=torch.float32), 
                torch.tensor(X_train_flat, dtype=torch.float32)
            )
            val_ae_dataset = TensorDataset(
                torch.tensor(X_val_flat, dtype=torch.float32), 
                torch.tensor(X_val_flat, dtype=torch.float32)
            )
            
            train_ae_loader = DataLoader(train_ae_dataset, batch_size=batch_size, shuffle=True)
            val_ae_loader = DataLoader(val_ae_dataset, batch_size=batch_size, shuffle=False)
            
            # 3. Train PyTorch LSTM Forecaster
            self.update_status(message="Training LSTM Forecaster (PyTorch on GPU)...")
            lstm_model = LSTMForecaster(feature_dim=n_features)
            lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)
            lstm_criterion = nn.MSELoss()
            
            # LSTM forecasts next step, so X predicts y
            train_lstm_dataset = TensorDataset(
                torch.tensor(X_train, dtype=torch.float32), 
                torch.tensor(y_train, dtype=torch.float32)
            )
            val_lstm_dataset = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32), 
                torch.tensor(y_val, dtype=torch.float32)
            )
            
            train_lstm_loader = DataLoader(train_lstm_dataset, batch_size=batch_size, shuffle=True)
            val_lstm_loader = DataLoader(val_lstm_dataset, batch_size=batch_size, shuffle=False)
            
            # Joint Training Loop
            ae_train_losses = []
            ae_val_losses = []
            
            start_time = time.time()
            for epoch in range(1, epochs + 1):
                epoch_start = time.time()
                
                # Training Autoencoder & LSTM
                autoencoder.train()
                lstm_model.train()
                
                train_loss_accum = 0.0
                batch_count = 0
                
                # Zip the two dataloaders
                for (ae_x, ae_y), (lstm_x, lstm_y) in zip(train_ae_loader, train_lstm_loader):
                    # Move to GPU
                    ae_x, ae_y = ae_x.to(device), ae_y.to(device)
                    lstm_x, lstm_y = lstm_x.to(device), lstm_y.to(device)
                    
                    # Autoencoder Step
                    ae_optimizer.zero_grad()
                    ae_pred = autoencoder(ae_x)
                    ae_loss = ae_criterion(ae_pred, ae_y)
                    ae_loss.backward()
                    ae_optimizer.step()
                    
                    # LSTM Step
                    lstm_optimizer.zero_grad()
                    lstm_pred = lstm_model(lstm_x)
                    lstm_loss = lstm_criterion(lstm_pred, lstm_y)
                    lstm_loss.backward()
                    lstm_optimizer.step()
                    
                    train_loss_accum += ae_loss.item() + lstm_loss.item()
                    batch_count += 1
                
                # Validation loop
                autoencoder.eval()
                lstm_model.eval()
                val_loss_accum = 0.0
                val_batch_count = 0
                
                with torch.no_grad():
                    for (ae_vx, ae_vy), (lstm_vx, lstm_vy) in zip(val_ae_loader, val_lstm_loader):
                        ae_vx, ae_vy = ae_vx.to(device), ae_vy.to(device)
                        lstm_vx, lstm_vy = lstm_vx.to(device), lstm_vy.to(device)
                        
                        ae_vpred = autoencoder(ae_vx)
                        ae_vloss = ae_criterion(ae_vpred, ae_vy)
                        
                        lstm_vpred = lstm_model(lstm_vx)
                        lstm_vloss = lstm_criterion(lstm_vpred, lstm_vy)
                        
                        val_loss_accum += ae_vloss.item() + lstm_vloss.item()
                        val_batch_count += 1
                        
                avg_train = train_loss_accum / max(1, batch_count)
                avg_val = val_loss_accum / max(1, val_batch_count)
                
                # Update stats
                ae_train_losses.append(avg_train)
                ae_val_losses.append(avg_val)
                
                epoch_dur = time.time() - epoch_start
                eta = int(epoch_dur * (epochs - epoch))
                
                self.update_status(
                    current_epoch=epoch,
                    train_loss=ae_train_losses,
                    val_loss=ae_val_losses,
                    training_speed=f"{epoch_dur:.2f}s/epoch",
                    eta_seconds=eta,
                    message=f"Epoch {epoch}/{epochs} - Loss: {avg_train:.4f}"
                )
                
            # Save final PyTorch weights
            ae_path = os.path.join(MODEL_DIR, f"{dataset_name}_autoencoder.pt")
            lstm_path = os.path.join(MODEL_DIR, f"{dataset_name}_lstm.pt")
            torch.save(autoencoder.state_dict(), ae_path)
            torch.save(lstm_model.state_dict(), lstm_path)
            
            # Cache active models
            with self.lock:
                self.active_models[f"{dataset_name}_forest"] = forest_model
                self.active_models[f"{dataset_name}_autoencoder"] = autoencoder
                self.active_models[f"{dataset_name}_lstm"] = lstm_model
                
            self.update_status(
                is_training=False,
                message="Training Complete! Models saved.",
                eta_seconds=0
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.update_status(
                is_training=False,
                message=f"Error during training: {str(e)}"
            )

    def load_cached_models(self, dataset_name, num_features):
        """Loads saved models from disk if not already cached in memory."""
        forest_path = os.path.join(MODEL_DIR, f"{dataset_name}_forest.pkl")
        ae_path = os.path.join(MODEL_DIR, f"{dataset_name}_autoencoder.pt")
        lstm_path = os.path.join(MODEL_DIR, f"{dataset_name}_lstm.pt")
        
        # Check files
        if not (os.path.exists(forest_path) and os.path.exists(ae_path) and os.path.exists(lstm_path)):
            return False
            
        try:
            # Forest
            if f"{dataset_name}_forest" not in self.active_models:
                self.active_models[f"{dataset_name}_forest"] = IsolationForestWrapper.load(forest_path)
                
            # Autoencoder
            if f"{dataset_name}_autoencoder" not in self.active_models:
                # 12 window * features
                ae = Autoencoder(input_dim=12 * num_features)
                ae.load_state_dict(torch.load(ae_path, map_location=ae.device))
                ae.eval()
                self.active_models[f"{dataset_name}_autoencoder"] = ae
                
            # LSTM
            if f"{dataset_name}_lstm" not in self.active_models:
                lstm = LSTMForecaster(feature_dim=num_features)
                lstm.load_state_dict(torch.load(lstm_path, map_location=lstm.device))
                lstm.eval()
                self.active_models[f"{dataset_name}_lstm"] = lstm
                
            return True
        except Exception as e:
            print(f"Failed to load cached models: {e}")
            return False

if __name__ == "__main__":
    print("Testing model trainer...")
    trainer = ModelTrainer()
    started, msg = trainer.start_training("industrial", epochs=3)
    print(msg)
    
    # Wait for completion
    while True:
        status = trainer.get_status()
        print(f"Status: {status['message']}")
        if not status["is_training"]:
            break
        time.sleep(1)
        
    # Check paths
    print("Check saved files:", os.listdir(MODEL_DIR))
