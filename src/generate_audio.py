import io
import torch
import base64
import torchaudio
import streamlit as st
import matplotlib.pyplot as plt
from model_GAN import Generator, DEVICE, NOISE_DIM

SR = 22050
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
TIME_DIM = 128
MODEL_PATH = "../models/Generator.pth"
NUM_SEGMENTS = 8

def load_model():
    model = Generator().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

G = load_model()

def generate_audio():
    segments = []
    mel_spectrograms = []

    mel_scale = torchaudio.transforms.MelScale(
        n_mels=N_MELS,
        sample_rate=SR,
        n_stft=513
    )
    mel_basis = mel_scale.fb
    pseudo_inv = torch.linalg.pinv(mel_basis.T)

    griffin = torchaudio.transforms.GriffinLim(
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_iter=100
    )

    with torch.no_grad():
        for _ in range(NUM_SEGMENTS):
            noise = torch.randn(1, NOISE_DIM, device=DEVICE)
            mel_tensor = G(noise).cpu().squeeze(0) 
            mel_spectrograms.append(mel_tensor)

            mel_db = mel_tensor * 80.0 - 80.0
            mel_amp = torchaudio.functional.DB_to_amplitude(mel_db, ref=1.0, power=2.0)
            mel_amp = torch.clamp(mel_amp, min=1e-5)

            spec_lin = torch.matmul(pseudo_inv, mel_amp)
            spec_lin = torch.clamp(spec_lin, min=0.0)
            spec_lin = spec_lin / (spec_lin.max() + 1e-8)

            audio = griffin(spec_lin.unsqueeze(0))
            audio = torch.nan_to_num(audio, nan=0.0)

            audio = audio * 50.0
            audio = torch.clamp(audio, -1.0, 1.0)

            segments.append(audio)

    full_audio = torch.cat(segments, dim=-1)
    return full_audio, mel_spectrograms[-1]

st.title("🎸 Metal Music Generator 🎸")

if st.button("Generate Audio"):
    st.info("Generating audio...")
    audio, mel_spectrogram = generate_audio()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(mel_spectrogram.squeeze(0).numpy(), aspect='auto', origin='lower', cmap='magma')
    ax.set_title("Generated Mel Spectrogram")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mel Frequency")
    st.pyplot(fig)


    buffer = io.BytesIO()
    torchaudio.save(buffer, audio.squeeze(0), SR, format="wav")
    buffer.seek(0)
    audio_bytes = buffer.read()

    st.audio(audio_bytes, format="audio/wav")
    b64 = base64.b64encode(audio_bytes).decode()
    href = f'<a href="data:audio/wav;base64,{b64}" download="generated_audio.wav">Download Audio</a>'
    st.markdown(href, unsafe_allow_html=True)
