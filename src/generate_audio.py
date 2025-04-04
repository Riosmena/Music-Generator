import torch
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from train_model import VAE

SR = 16000
N_MELS = 128
LATENT_DIM = 32
LENGTH = 940
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VAE((1, N_MELS, LENGTH), LATENT_DIM).to(DEVICE)
model.load_state_dict(torch.load("vae_model.pth"))
model.eval()

z = torch.randn(1, LATENT_DIM).to(DEVICE)

with torch.no_grad():
    generated = model.decode(z).cpu().squeeze().numpy()

generated = np.clip(generated, 0, 1)
mel_db = generated * 80.0 - 80.0

mel_power = librosa.db_to_power(mel_db)

audio = librosa.feature.inverse.mel_to_audio(
    mel_power, 
    sr=SR, 
    n_fft=2048, 
    hop_length=512, 
    win_length=2048, 
    n_iter=32,
    power=2.0
)

sf.write("generated.wav", audio, SR)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_db, sr=SR, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Generated Mel Spectrogram')
plt.tight_layout()
plt.show()