import torch
import torchaudio
import matplotlib.pyplot as plt
from model_GAN import Generator, DEVICE, NOISE_DIM, MusicDataset

SR = 22050
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
DATASET_PATH = "processed"
MODEL_PATH = "generator.pth"
OUTPUT_WAV_PATH = "metal_generated.wav"

dataset = MusicDataset(DATASET_PATH)
input_shape = dataset[0].shape

G = Generator(NOISE_DIM, input_shape).to(DEVICE)
G.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
G.eval()

noise = torch.randn(1, NOISE_DIM).to(DEVICE)
with torch.no_grad():
    generated = G(noise)

generated = (generated + 1) / 2

mel_spec_db = generated.squeeze(0).cpu() * 80.0 - 80.0

mel_spec_amp = torch.pow(10.0, mel_spec_db / 20.0)

mel_scale = torchaudio.transforms.MelScale(
    n_mels=N_MELS,
    sample_rate=SR,
    n_stft=N_FFT // 2 + 1,
    f_min=0.0,
    f_max=8000.0,
    norm="slaney"
)

mel_basis = mel_scale.fb
inv_mel_basis = torch.linalg.pinv(mel_basis.T)

mel_spec_amp = mel_spec_amp.float()
linear_spec = torch.matmul(inv_mel_basis, mel_spec_amp)
linear_spec = torch.clamp(linear_spec, min=1e-5, max=1e3)
linear_spec = linear_spec / (linear_spec.max() + 1e-6)

griffin = torchaudio.transforms.GriffinLim(n_fft=N_FFT, hop_length=HOP_LENGTH)
audio = griffin(linear_spec)
audio = torch.nan_to_num(audio, nan=0.0)

torchaudio.save("metal_generated.wav", audio.unsqueeze(0) if audio.ndim == 1 else audio, SR)
print(f"Audio saved to 'metal_generated.wav'")

plt.imshow(generated.squeeze().cpu(), aspect='auto', origin='lower', cmap='magma')
plt.title(f"Generated Spectrogram - Metal")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.show()