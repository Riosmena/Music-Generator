import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

DATASET_DIR = "data"
OUTPUT_DIR = "processed"
SAMPLE_RATE = 16000
DURATION = 30

def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def preprocess_audio_file(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True, duration=DURATION)
    return y

def save_spectrogram(audio, sr, output_path, target_length=940):
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)

    S_DB = (S_DB + 80.0) / 80.0
    S_DB = np.clip(S_DB, 0, 1)
    
    if S_DB.shape[1] < target_length:
        path_width = target_length - S_DB.shape[1]
        S_DB = np.pad(S_DB, ((0, 0), (0, path_width)), mode='constant')

    elif S_DB.shape[1] > target_length:
        S_DB = S_DB[:, :target_length]

    np.save(output_path, S_DB)

def process_dataset():
    ensure_dir_exists(OUTPUT_DIR)
    genres = os.listdir(DATASET_DIR)

    for genre in genres:
        genre_path = os.path.join(DATASET_DIR, genre)
        if not os.path.isdir(genre_path):
            continue
        
        genre_output_path = os.path.join(OUTPUT_DIR, genre)
        ensure_dir_exists(genre_output_path)

        for i, file in enumerate(os.listdir(genre_path)):
            if not file.endswith(".wav") and not file.endswith(".mp3"):
                continue

            file_path = os.path.join(genre_path, file)
            try:
                audio = preprocess_audio_file(file_path)
                output_path = os.path.join(genre_output_path, f"{genre}_{i}.npy")
                save_spectrogram(audio, SAMPLE_RATE, output_path)
            except:
                print(f"Error processing {file_path}")

process_dataset()
