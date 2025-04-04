import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

PROCESSED_DATA_DIR = "processed"
BATCH_SIZE = 16
EPOCHS = 30
LATENT_DIM = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SpectrogramDataset(Dataset):
    def __init__(self, base_path):
        self.data = []
        self.labels = []
        self.genre_to_idx = {}
        genres = os.listdir(base_path)

        for idx, genre in enumerate(genres):
            self.genre_to_idx[genre] = idx
            genre_path = os.path.join(base_path, genre)
            for file in os.listdir(genre_path):
                if file.endswith(".npy"):
                    self.data.append(os.path.join(genre_path, file))
                    self.labels.append(idx)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        spectrogram = np.load(self.data[idx])
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx])
        return spectrogram, label
    
class VAE(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(VAE, self).__init__()
        C, H, W = input_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(C, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(16 * (H // 2) * (W // 2), latent_dim)
        self.fc_logvar = nn.Linear(16 * (H // 2) * (W // 2), latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, 16 * (H // 2) * (W // 2))
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (16, H // 2, W // 2)),
            nn.ConvTranspose2d(16, C, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.decoder_fc(z)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
def vae_loss(recon_x, x, mu, logvar):
    recon_x = torch.clamp(recon_x, min=0, max=1)
    x = torch.clamp(x, min=0, max=1)

    bce = nn.functional.mse_loss(recon_x, x, reduction='mean')
    logvar = torch.clamp(logvar, min=-10, max=10)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return bce + kld

def train_vae():
    dataset = SpectrogramDataset(PROCESSED_DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    sample_input, _ = dataset[0]
    input_shape = sample_input.shape
    model = VAE(input_shape, LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in loader:
            x, _ = batch
            x = x.to(DEVICE)
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss:.2f}")

    torch.save(model.state_dict(), "vae_model.pth")

train_vae()