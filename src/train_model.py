import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.autograd as autograd
import matplotlib.pyplot as plt
import torchaudio
from model_GAN import MusicDataset, Generator, Discriminator, NOISE_DIM, DEVICE, TIME_DIM, N_MELS

# Paths & hyperparams
DATASET_PATH = "processed"
OUTPUT_DIR = "models"
SAMPLES_DIR = "samples"
EPOCHS = 300
BATCH_SIZE = 8
SAVE_EVERY = 5
LRG = 2e-4
LRD = 1e-4
BETA1, BETA2 = 0.5, 0.999
LAMBDA_GP = 5
MARGIN = 0.3

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

def save_checkpoint(model, epoch):
    model.eval()
    with torch.no_grad():
        noise = torch.randn(1, NOISE_DIM, device=DEVICE)
        mel_tensor = model(noise).cpu().squeeze(0)

    mel_db = mel_tensor * 80.0 - 80.0
    mel_amp = torchaudio.functional.DB_to_amplitude(mel_db, ref=1.0, power=2.0)
    mel_amp = torch.clamp(mel_amp, min=1e-5)

    mel_scale = torchaudio.transforms.MelScale(
            n_stft=513,
            n_mels=N_MELS,
            sample_rate=22050,
        )

    mel_basis = mel_scale.fb
    pseudo_inv = torch.linalg.pinv(mel_basis.T)
    spec_lin = torch.matmul(pseudo_inv, mel_amp)

    spec_lin = torch.clamp(spec_lin, min=0.0)
    spec_lin = spec_lin / (spec_lin.max() + 1e-8)

    griffin = torchaudio.transforms.GriffinLim(
        n_fft=1024,
        hop_length=512,
        n_iter=100
    )
    audio = griffin(spec_lin.unsqueeze(0))
    audio = torch.nan_to_num(audio, nan=0.0)

    audio = audio * 5.0
    audio = torch.clamp(audio, -1.0, 1.0)

    wav_path = os.path.join(SAMPLES_DIR, f"sample_epoch_{epoch}.wav")
    torchaudio.save(wav_path, audio.squeeze(0), 22050)

    plt.figure(figsize=(10, 4))
    plt.imshow(mel_tensor.squeeze(0), aspect='auto', origin='lower', cmap='magma')
    plt.title(f"Generated Mel Spectrogram (Epoch {epoch})")
    plt.xlabel("Time") 
    plt.ylabel("Mel Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(SAMPLES_DIR, f"sample_epoch_{epoch}.png"))
    plt.close()
    model.train()

def gradient_penalty(D, real, fake, device):
    batch_size = real.size(0)
    eps = torch.rand(batch_size, 1, 1, 1, device=device)
    interp = eps * real + (1 - eps) * fake
    interp.requires_grad_(True)
    d_interp = D(interp)
    grads = autograd.grad(
        outputs=d_interp, inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
    )[0]
    grad_norm = grads.view(batch_size, -1).norm(2, dim=1)
    return ((grad_norm - 1) ** 2).mean()

def train():
    dataset = MusicDataset(DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)

    opt_g = optim.Adam(G.parameters(), lr=LRG, betas=(BETA1, BETA2))
    opt_d = optim.Adam(D.parameters(), lr=LRD, betas=(BETA1, BETA2))
    sched_g = StepLR(opt_g, step_size=50, gamma=0.5)
    sched_d = StepLR(opt_d, step_size=50, gamma=0.5)

    losses_d = []
    losses_g = []

    for epoch in range(1, EPOCHS+1):
        for real in dataloader:
            real = real.to(DEVICE)
            b_size = real.size(0)

            noise = torch.randn(b_size, NOISE_DIM, device=DEVICE)
            fake = G(noise)
            d_real = D(real).mean()
            d_fake = D(fake.detach()).mean()
            gp = gradient_penalty(D, real, fake, DEVICE)
            loss_d = d_fake - d_real + LAMBDA_GP * gp

            advantage = (d_real - d_fake).item()

            if advantage < MARGIN:
                D.zero_grad()
                loss_d.backward()
                opt_d.step()

                for param_group in opt_d.param_groups:
                    if param_group['lr'] < LRD:
                        param_group['lr'] = LRD
            else:
                for param_group in opt_d.param_groups:
                    param_group['lr'] = LRD * 0.25

            noise = torch.randn(b_size, NOISE_DIM, device=DEVICE)
            fake = G(noise)
            loss_g = -D(fake).mean()

            real_score = d_real.item()
            fake_score = D(fake).mean().item()

            G.zero_grad()
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            opt_g.step()

        sched_g.step()
        sched_d.step()

        losses_d.append(loss_d.item())
        losses_g.append(loss_g.item())

        print(f"[Epoch {epoch}/{EPOCHS}] Loss_D: {loss_d.item():.4f}, Loss_G: {loss_g.item():.4f}, Real: {real_score:.4f}, Fake: {fake_score:.4f}")
        if epoch % SAVE_EVERY == 0:
            torch.save(G.state_dict(), os.path.join(OUTPUT_DIR, f"Generator.pth"))
            save_checkpoint(G, epoch)


    plt.figure(figsize=(10, 4))
    plt.plot(losses_d, label='Loss D', color='red')
    plt.plot(losses_g, label='Loss G', color='green')
    plt.title('Losses over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(SAMPLES_DIR, "losses.png"))
    plt.show()

if __name__ == "__main__":
    torch.cuda.empty_cache()
    train()
