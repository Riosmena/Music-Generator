import os
import torch
import torchaudio
from tqdm import tqdm

GENRE = "metal"
INPUT_DIR = os.path.join("data", GENRE)
OUTPUT_DIR = os.path.join("processed", GENRE)
SR = 22050
CHUNK_DURATION = 6
N_MELS = 64
SAMPLES = SR *CHUNK_DURATION

os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_audio(file_path):
    signal, sr = torchaudio.load(file_path)
    signal = torch.mean(signal, dim=0, keepdim=True)

    total_samples = signal.shape[1]
    chunks = total_samples // SAMPLES
    processed = []

    for i in range(chunks):
        start = i * SAMPLES
        end = start + SAMPLES
        chunk = signal[:, start:end]

        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR,
            n_mels=N_MELS
        )(chunk)
        mel_db = torchaudio.transforms.AmplitudeToDB()(mel)
        processed.append(mel_db)
    
    return processed

def process_dataset():
    for file in tqdm(os.listdir(INPUT_DIR), desc="Processing files", colour="green"):
        if file.endswith(".wav"):
            in_path = os.path.join(INPUT_DIR, file)
            chunks = process_audio(in_path)
            base_name = os.path.splitext(file)[0]
            for idx, chunk in enumerate(chunks):
                out_path = os.path.join(OUTPUT_DIR, f"{base_name}_{idx:03d}.pt")
                torch.save(chunk, out_path)

process_dataset()
