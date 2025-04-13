import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from glob import glob

NOISE_DIM = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MusicDataset(Dataset):
    def __init__(self, data_dir):
        self.genre = "metal"
        self.samples = glob(os.path.join(data_dir, self.genre, "*.pt"))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        spec= torch.load(self.samples[idx])
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)
        spec = spec * 2 - 1

        if spec.ndim == 2:
            return spec.unsqueeze(0)
        elif spec.ndim == 3 and spec.shape[0] == 1:
            return spec
    
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(100, 512, kernel_size=(4, 4), stride=(1, 1), padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 1, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, kernel_size=(4, 4), stride=(1, 1), padding=0),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
    
    def forward(self, x):
        return self.net(x)
    