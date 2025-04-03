import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

DATA_PATH = "data"
SAMPLE_RATE = 22050
DURATION = 10
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
N_MELS = 128
LATENT_DIM = 128
BATCH_SIZE = 32
EPOCHS = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

def get_genre_encodings(genres):
    le = LabelEncoder()
    int_labels = le.fit_transform(genres)
    ohe = OneHotEncoder(sparse_output=False)
    onehots = ohe.fit_transform(int_labels.reshape(-1, 1))
    return le, {genre: onehots[i] for i, genre in enumerate(genres)}

def preprocess_audio(data_path):
    genres = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    label_encoder, genre_dict = get_genre_encodings(genres)
    X, y = [], []

    for genre in genres:
        genre_path = os.path.join(data_path, genre)
        print(f"Processing genre: {genre}")
        for file in os.listdir(genre_path):
            if file.endswith(".wav"):
                try:
                    filepath = os.path.join(genre_path, file)
                    signal, sr = librosa.load(filepath, sr=SAMPLE_RATE)
                    if len(signal) >= SAMPLES_PER_TRACK:
                        signal = signal[:SAMPLES_PER_TRACK]
                        mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=N_MELS)
                        mel_db = librosa.power_to_db(mel, ref=np.max)
                        mel_db = mel_db / 80.0 + 1.0  # Normaliza a [0,1]
                        X.append(mel_db)
                        y.append(genre_dict[genre])
                except:
                    print(f"Error with {file}")
    X = np.array(X)[..., np.newaxis]
    y = np.array(y)
    return X, y, label_encoder

class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2) 
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class Sampling(nn.Module):
    def forward(self, z_mean, z_log_var):
        epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

class Encoder(nn.Module):
    def __init__(self, genre_dim, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc_genre = nn.Linear(genre_dim, 64 * 32 * 108)
        self.fc = nn.Linear(64 * 32 * 108 * 2, 256)
        self.z_mean = nn.Linear(256, latent_dim)
        self.z_log_var = nn.Linear(256, latent_dim)
        self.sampling = Sampling()

    def forward(self, x, genre):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        g = F.relu(self.fc_genre(genre))
        concat = torch.cat((x, g), dim=1)
        h = F.relu(self.fc(concat))
        z_mean = self.z_mean(h)
        z_log_var = self.z_log_var(h)
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z

class Decoder(nn.Module):
    def __init__(self, genre_dim, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim + genre_dim, 128 * 431)
        self.unflatten = nn.Unflatten(1, (1, 128, 431))
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, z, genre):
        x = torch.cat((z, genre), dim=1)
        x = F.relu(self.fc(x))
        x = self.unflatten(x)
        x = F.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        return x

def train_vae(encoder, decoder, dataloader, optimizer, kl_weight=0.1):
    encoder.train()
    decoder.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            z_mean, z_log_var, z = encoder(batch_x, batch_y)
            recon = decoder(z, batch_y)
            recon_loss = F.mse_loss(recon, batch_x)
            kl_loss = -0.5 * torch.mean(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
            loss = recon_loss + kl_weight * kl_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(dataloader):.4f}")

def spectrogram_to_audio(mel, output_file="output.wav"):
    mel = mel.squeeze()
    mel = mel.cpu().detach().numpy()
    mel = np.clip(mel, 0, 1)
    mel_db = mel * 80.0
    mel_power = librosa.db_to_power(mel_db)
    stft = librosa.feature.inverse.mel_to_stft(mel_power, sr=SAMPLE_RATE)
    audio = librosa.griffinlim(stft)
    sf.write(output_file, audio, SAMPLE_RATE)
    print(f"Saved generated audio in: {output_file}")

if __name__ == "__main__":
    X, y, label_encoder = preprocess_audio(DATA_PATH)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = AudioDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    genre_dim = y.shape[1]

    encoder = Encoder(genre_dim, LATENT_DIM).to(device)
    decoder = Decoder(genre_dim, LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)

    train_vae(encoder, decoder, train_loader, optimizer)

    genre_name = label_encoder.classes_[6]
    genre_index = list(label_encoder.classes_).index(genre_name)
    genre_onehot = np.zeros((1, genre_dim))
    genre_onehot[0, genre_index] = 1

    z_random = torch.randn(1, LATENT_DIM).to(device)
    genre_tensor = torch.tensor(genre_onehot, dtype=torch.float32).to(device)

    with torch.no_grad():
        generated = decoder(z_random, genre_tensor)
        spectrogram = generated.squeeze()

    plt.imshow(spectrogram.cpu(), origin='lower', aspect='auto', cmap='magma')
    plt.title(f"Generated genre: {genre_name}")
    plt.colorbar()
    plt.show()

    spectrogram_to_audio(spectrogram, f"{genre_name}_generated.wav")
