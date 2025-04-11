import torch
import torchaudio
import matplotlib.pyplot as plt
from model_GAN import Generator, DEVICE, NOISE_DIM, MusicDataset

MODEL_PATH = "generator.pth"
DATASET_PATH = "processed"
OUTPUT_SPEC_PATH = "generated_spectrogram.pt"
OUTPUT_AUDIO = "generated_audio.wav"
SR = 22050
N_MELS = 64

dataset = MusicDataset(DATASET_PATH)
genre_dim = len(dataset.genres)
_, _genre = dataset[0]
input_shape = _.shape

generator = Generator(NOISE_DIM, genre_dim, input_shape).to(DEVICE)
generator.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
generator.eval()

genre_names = dataset.genres
print("Availables Genres:", genre_names)
genre_idx = int(input(f"Select a genre index (0-{len(genre_names) - 1}): "))
genre_onehot = torch.zeros(1, genre_dim).to(DEVICE)
genre_onehot[0, genre_idx] = 1.0

noise = torch.randn(1, NOISE_DIM).to(DEVICE)
with torch.no_grad():
    generated = generator(noise, genre_onehot)

generated = (generated + 1) / 2
generated = generated.cpu().squeeze(0)
torch.save(generated, OUTPUT_SPEC_PATH)
print(f"Spectrogram saved to {OUTPUT_SPEC_PATH}")

mel_spec_db = generated * 80.0 - 80.0
mel_spec_amp = torch.pow(10.0, mel_spec_db / 20.0)

griffin_lim = torchaudio.transforms.GriffinLim(n_fft=1024)
audio = griffin_lim(mel_spec_amp)

torchaudio.save(OUTPUT_AUDIO, audio.unsqueeze(0), SR)
print(f"Audio saved to {OUTPUT_AUDIO}")

plt.figure(figsize=(10, 4))
plt.imshow(generated, aspect='auto', origin='lower', cmap='magma')
plt.title(f"Generated Spectrogram - Genre: {genre_names[genre_idx]}")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.tight_layout()
plt.show()