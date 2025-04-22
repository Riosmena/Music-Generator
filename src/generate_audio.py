import torch
import torchaudio
import matplotlib.pyplot as plt
from model_GAN import Generator, DEVICE, NOISE_DIM

SR = 22050
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
TIME_DIM = 128
MODEL_PATH = "logs/G_epoch200.pth"
OUTPUT_WAV = "metal_generated.wav"
NUM_SEGMENTS = 8

G = Generator().to(DEVICE)
G.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
G.eval()
segments = []

with torch.no_grad():
    for _ in range(NUM_SEGMENTS):
        noise = torch.randn(1, NOISE_DIM, device=DEVICE)
        generated = G(noise).cpu().squeeze(0)

        mel_db = generated * 80.0 - 80.0

        mel_amp = torchaudio.functional.DB_to_amplitude(mel_db, ref=1.0, power=0.5)

        inv_mel = torchaudio.transforms.InverseMelScale(
            n_stft=N_FFT // 2 + 1,
            n_mels=N_MELS,
            sample_rate=SR,
        )

        spec_lin = inv_mel(mel_amp)

        griffin = torchaudio.transforms.GriffinLim(
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_iter=100
        )

        audio = griffin(spec_lin)
        audio = torch.nan_to_num(audio, nan=0.0)
        segments.append(audio)

full = torch.cat(segments, dim=-1)
torchaudio.save(OUTPUT_WAV, full.unsqueeze(0), SR)
print(f"Generated audio saved to {OUTPUT_WAV}")

plt.figure(figsize=(10, 4))
plt.imshow(generated.squeeze(), aspect='auto', origin='lower', cmap='magma')
plt.title("Generated Mel Spectrogram")
plt.xlabel("Time")
plt.ylabel("Mel Frequency")
plt.tight_layout()
plt.show()