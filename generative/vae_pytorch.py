"""
Variational Autoencoder (VAE) implementation in PyTorch.
Covers: Encoder, Decoder, and the Reparameterization Trick.

Math:
Loss = Reconstruction_Loss + KL_Divergence
KL_Divergence = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, h_dim)
        self.fc2_mu = nn.Linear(h_dim, z_dim)    # Mean
        self.fc2_logvar = nn.Linear(h_dim, z_dim) # Variance
        
        # Decoder
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2_mu(h), self.fc2_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std # z = mu + eps * sigma

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

if __name__ == "__main__":
    # Test model
    model = VAE()
    dummy_x = torch.randn(1, 784)
    recon_x, mu, logvar = model(dummy_x)
    
    loss = vae_loss_function(recon_x, dummy_x, mu, logvar)
    print(f"Reconstruction Shape: {recon_x.shape}")
    print(f"Loss Value: {loss.item():.4f}")
    print("VAE Forward Pass Successful.")
