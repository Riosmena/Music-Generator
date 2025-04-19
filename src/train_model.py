import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.autograd as autograd
from model_GAN import MusicDataset, Generator, Discriminator, NOISE_DIM, DEVICE, TIME_DIM

# Paths & hyperparams
DATASET_PATH = "processed"
OUTPUT_DIR = "logs"
EPOCHS = 200
BATCH_SIZE = 8
SAVE_EVERY = 10
LR = 2e-4
BETA1, BETA2 = 0.5, 0.9
LAMBDA_GP = 10
CRITIC_ITER = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

def gradient_penalty(D, real, fake, device):
    batch_size = real.size(0)
    eps = torch.rand(batch_size, 1, 1, 1, device=device)
    interp = eps * real + (1 - eps) * fake
    interp.requires_grad_(True)
    d_interp = D(interp)
    grads = autograd.grad(
        outputs=d_interp, inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True
    )[0]
    grad_norm = grads.view(batch_size, -1).norm(2, dim=1)
    return ((grad_norm - 1) ** 2).mean()

def train():
    dataset = MusicDataset(DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)

    opt_g = optim.Adam(G.parameters(), lr=LR, betas=(BETA1, BETA2))
    opt_d = optim.Adam(D.parameters(), lr=LR, betas=(BETA1, BETA2))
    sched_g = StepLR(opt_g, step_size=50, gamma=0.5)
    sched_d = StepLR(opt_d, step_size=50, gamma=0.5)

    for epoch in range(1, EPOCHS+1):
        for real in dataloader:
            real = real.to(DEVICE)
            b_size = real.size(0)

            # Train Discriminator (Critic)
            for _ in range(CRITIC_ITER):
                noise = torch.randn(b_size, NOISE_DIM, device=DEVICE)
                fake = G(noise)
                d_real = D(real).mean()
                d_fake = D(fake.detach()).mean()
                gp = gradient_penalty(D, real, fake, DEVICE)
                loss_d = d_fake - d_real + LAMBDA_GP * gp

                D.zero_grad()
                loss_d.backward()
                opt_d.step()

            # Train Generator
            noise = torch.randn(b_size, NOISE_DIM, device=DEVICE)
            fake = G(noise)
            loss_g = -D(fake).mean()

            G.zero_grad()
            loss_g.backward()
            opt_g.step()

        sched_g.step()
        sched_d.step()

        print(f"[Epoch {epoch}/{EPOCHS}] Loss_D: {loss_d.item():.4f}, Loss_G: {loss_g.item():.4f}")
        if epoch % SAVE_EVERY == 0:
            torch.save(G.state_dict(), os.path.join(OUTPUT_DIR, f"G_epoch{epoch}.pth"))

if __name__ == "__main__":
    torch.cuda.empty_cache()
    train()
