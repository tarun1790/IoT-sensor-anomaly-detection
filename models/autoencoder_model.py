import torch
import torch.nn as nn
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=4):
        super(Autoencoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Autoencoder model initialized on device: {self.device}")
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
        
        self.to(self.device)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
        
    def reconstruct(self, x_np):
        """
        x_np: numpy array of shape (batch, input_dim)
        Returns: reconstructed numpy array of shape (batch, input_dim)
        """
        self.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x_np, dtype=torch.float32).to(self.device)
            reconstructed = self.forward(x_tensor)
            return reconstructed.cpu().numpy()

    def get_reconstruction_error(self, x_np):
        """
        Computes reconstruction error per feature.
        """
        x_recon = self.reconstruct(x_np)
        # Returns squared error for each element
        return (x_np - x_recon) ** 2
