import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from model_GAN import MusicDataset, Generator, Discriminator, NOISE_DIM, DEVICE

DATASET_PATH = "processed"
EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 0.0002
    
def train():
    dataset = MusicDataset(DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    genre_dim = len(dataset.genres)
    example_spec, _ = dataset[0]
    input_shape = example_spec.shape

    generator = Generator(NOISE_DIM, genre_dim, input_shape).to(DEVICE)
    discriminator = Discriminator(genre_dim, input_shape).to(DEVICE)

    criterion = nn.BCELoss()
    opt_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    print("Beginning training...")

    for epoch in range(EPOCHS):
        for real_specs, genres in dataloader:
            batch_size = real_specs.size(0)
            real_specs, genres = real_specs.to(DEVICE), genres.to(DEVICE)

            real_specs += 0.05 * torch.randn_like(real_specs)

            real_labels = torch.full((batch_size, 1), 0.9).to(DEVICE)
            fake_labels = torch.zeros(batch_size, 1).to(DEVICE)

            noise = torch.randn(batch_size, NOISE_DIM).to(DEVICE)
            fake_specs = generator(noise, genres)

            d_real = discriminator(real_specs, genres)
            d_fake = discriminator(fake_specs.detach(), genres)

            loss_d_real = criterion(d_real, real_labels)
            loss_d_fake = criterion(d_fake, fake_labels)
            loss_d = loss_d_real + loss_d_fake

            discriminator.zero_grad()
            loss_d.backward()
            opt_d.step()

            loss_g = criterion(discriminator(fake_specs, genres), real_labels)

            generator.zero_grad()
            loss_g.backward()
            opt_g.step()
    
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss D: {loss_d.item()}, Loss G: {loss_g.item()}")
    
    torch.save(generator.state_dict(), "generator.pth")
    torch.save(discriminator.state_dict(), "discriminator.pth")
    print("Models saved")

torch.cuda.empty_cache()
train()
