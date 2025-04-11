import os
import librosa
import torch
import torchaudio
from torchvision.transforms import Resize
from tqdm import tqdm

DATASET = "data"
PROCESSED_DATA_DIR = "processed"
SR = 22050
DURATION = 10
SAMPLES_PER_TRACK = SR * DURATION
N_MELS = 64

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def process_audio(file_path):
    signal, sr = torchaudio.load(file_path)
    signal = torch.mean(signal, dim=0, keepdim=True)

    if signal.shape[1] > SAMPLES_PER_TRACK:
        signal = signal[:, :SAMPLES_PER_TRACK]

    else:
        pad = SAMPLES_PER_TRACK - signal.shape[1]
        signal = torch.nn.functional.pad(signal, (0, pad), "constant")

    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_mels=N_MELS,
    )(signal)

    mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    return mel_spec.squeeze(0)

def process_dataset():
    for genre in os.listdir(DATASET):
        genre_path = os.path.join(DATASET, genre)
        if not os.path.isdir(genre_path):
            continue

        genre_out = os.path.join(PROCESSED_DATA_DIR, genre)
        os.makedirs(genre_out, exist_ok=True)

        for file in tqdm(os.listdir(genre_path), desc=f"Processing {genre}"):
            if not file.endswith(".wav"):
                continue

            file_path = os.path.join(genre_path, file)
            try:
                mel = process_audio(file_path)
                out_path = os.path.join(genre_out, file.replace(".wav", ".pt"))
                torch.save(mel, out_path)
            except:
                print(f"Error processing {file_path}")
                continue

process_dataset()