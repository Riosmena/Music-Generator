import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset
from glob import glob

NOISE_DIM = 100
TIME_DIM = 64
N_MELS = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MusicDataset(Dataset):
    def __init__(self, data_dir):
        self.base = data_dir
        self.samples = glob(os.path.join(self.base, "metal", "*.pt"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        spec = torch.load(self.samples[idx], weights_only=True)
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)
        spec = spec * 2 - 1

        # Ensure dims [1, N_MELS, T]
        if spec.ndim == 2:
            spec = spec.unsqueeze(0)
        # Resize to fixed [1, N_MELS, TIME_DIM]
        spec = spec.unsqueeze(0)                      # [1,1,N_MELS,T]
        spec = F.interpolate(
            spec, size=(N_MELS, TIME_DIM),
            mode='bilinear', align_corners=False
        )                                             # [1,1,N_MELS,TIME_DIM]
        spec = spec.squeeze(0)                        # [1,N_MELS,TIME_DIM]
        return spec

class Generator(nn.Module):
    def __init__(self, noise_dim=NOISE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(z.size(0), -1, 1, 1)
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(1, 64, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(512, 1, 4, 1, 0, bias=False)),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, x):
        return self.net(x).view(-1)
