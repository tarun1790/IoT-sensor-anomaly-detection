import torch
import torch.nn as nn
import numpy as np

class LSTMForecaster(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64, num_layers=2):
        super(LSTMForecaster, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"LSTM model initialized on device: {self.device}")
        
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, feature_dim)
        self.to(self.device)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, (hn, cn) = self.lstm(x)
        # Predict based on the last sequence element's hidden state
        predictions = self.fc(out[:, -1, :])
        return predictions
        
    def predict(self, x_np):
        """
        x_np: numpy array of shape (batch, seq_len, features)
        Returns: predicted values of shape (batch, features)
        """
        self.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x_np, dtype=torch.float32).to(self.device)
            preds = self.forward(x_tensor)
            return preds.cpu().numpy()
            
    def get_prediction_error(self, x_np, y_np):
        """
        Computes prediction squared error.
        x_np: (batch, seq_len, features)
        y_np: (batch, features)
        """
        preds = self.predict(x_np)
        return (y_np - preds) ** 2
