import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from glob import glob

NOISE_DIM = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MusicDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        self.genres = sorted(os.listdir(data_dir))
        self.genre_to_idx = {g: i for i, g in enumerate(self.genres)}

        for genre in self.genres:
            genre_dir = os.path.join(data_dir, genre)
            for file in glob(f"{genre_dir}/*.pt"):
                self.samples.append((file, self.genre_to_idx[genre]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, genre_idx = self.samples[idx]
        spectrogram = torch.load(file_path)
        spectrogram = spectrogram.unsqueeze(0)
        spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min() + 1e-6)
        spectrogram = spectrogram * 2 - 1
        genre_onehot = torch.zeros(len(self.genres))
        genre_onehot[genre_idx] = 1.0
        return spectrogram, genre_onehot
    
class Generator(nn.Module):
    def __init__(self, noise_dim, genre_dim, output_shape):
        super().__init__()
        self.input_dim = noise_dim + genre_dim
        self.output_shape = output_shape

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Linear(2048, output_shape[1] * output_shape[2]),
            nn.Tanh(),
        )

    def forward(self, noise, genre):
        x = torch.cat([noise, genre], dim=1)
        x = self.net(x)
        return x.view(-1, *self.output_shape)

class Discriminator(nn.Module):
    def __init__(self, genre_dim, input_shape):
        super().__init__()
        self.input_dim = input_shape[1] * input_shape[2] + genre_dim

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, spectrogram, genre):
        x = spectrogram.view(spectrogram.size(0), -1)
        x = torch.cat([x, genre], dim=1)
        x = self.net(x)
        return x