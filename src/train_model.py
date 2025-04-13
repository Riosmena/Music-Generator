import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchaudio
import matplotlib.pyplot as plt
from model_GAN import MusicDataset, Generator, Discriminator, NOISE_DIM, DEVICE

DATASET_PATH = "processed"
OUTPUT_DIR = "logs"
EPOCHS = 200
BATCH_SIZE = 8
SAVE_EVERY = 10
LEARNING_RATE = 0.0002

os.makedirs(OUTPUT_DIR, exist_ok=True)

losses_G = []
losses_D = []

def save_generated_sample(generator, step):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(1, NOISE_DIM, 1, 1).to(DEVICE)
        fake = generator(noise).cpu().squeeze(0)

        fake = F.interpolate(fake.unsqueeze(0), size=(64, 662), mode='bilinear', align_corners=False).squeeze(0)

        img = (fake + 1) / 2
        plt.figure(figsize=(10, 4))
        plt.imshow(img.squeeze(0), aspect='auto', origin='lower', cmap='magma')
        plt.colorbar()
        plt.title(f"Generated Spectrogram - Epoch {step}")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/sample_{step}.png")
        plt.close()

        mel_spec = img.squeeze(0)
        mel_spec_db = mel_spec * 80.0 - 80.0
        mel_spec_amp = torch.pow(10.0, mel_spec_db / 20.0)

        mel_scale = torchaudio.transforms.MelScale(
            n_mels=64,
            sample_rate=22050,
            n_stft=512 + 1
        )
        mel_basis = mel_scale.fb
        inv_mel_basis = torch.linalg.pinv(mel_basis.T)
        linear_spec = torch.matmul(inv_mel_basis, mel_spec_amp)
        linear_spec = torch.clamp(linear_spec, min=1e-5)
        linear_spec = linear_spec / (linear_spec.max() + 1e-6)

        griffin = torchaudio.transforms.GriffinLim(n_fft=1024, hop_length=512)
        audio = griffin(linear_spec)
        audio = torch.nan_to_num(audio, nan=0.0)
        
        torchaudio.save(f"{OUTPUT_DIR}/sample_{step}.wav", audio.unsqueeze(0) if audio.ndim == 1 else audio, 22050)
    
    generator.train()
    
def train():
    dataset = MusicDataset(DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)

    opt_g = optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_d = optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    print("Beginning training...")

    for epoch in range(EPOCHS):
        for real in dataloader:
            real = real.to(DEVICE)
            batch_size = real.size(0)

            noise = torch.randn(batch_size, NOISE_DIM, 1, 1).to(DEVICE)
            fake = G(noise)

            d_real = D(real)
            d_fake = D(fake.detach())

            real_labels = torch.ones_like(d_real) * 0.9
            fake_labels = torch.zeros_like(d_fake)

            loss_d_real = criterion(d_real, real_labels)
            loss_d_fake = criterion(d_fake, fake_labels)
            loss_d = loss_d_real + loss_d_fake

            D.zero_grad()
            loss_d.backward()
            opt_d.step()

            d_fake_for_g = D(fake)
            loss_g = criterion(d_fake_for_g, real_labels)

            G.zero_grad()
            loss_g.backward()
            opt_g.step()

        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}")
        losses_G.append(loss_g.item())
        losses_D.append(loss_d.item())

        if epoch % SAVE_EVERY == 0:
            save_generated_sample(G, epoch)
            torch.save(G.state_dict(), "generator.pth")

    print("Model saved")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS + 1), losses_G, label='Generator Loss')
    plt.plot(range(1, EPOCHS + 1), losses_D, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/losses.png")
    plt.close()

torch.cuda.empty_cache()
train()
