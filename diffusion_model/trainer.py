"""Entraînement du modèle de diffusion."""
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from .noise import q_sample, sample_timesteps


class Trainer:
    def __init__(self, model, dataloader, lr=2e-4, device="cpu"):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.optim = Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def train(self, epochs=10, timesteps=1000):
        self.model.train()
        for epoch in range(epochs):
            loop = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in loop:
                if isinstance(batch, (tuple, list)):
                    x, _ = batch
                else:
                    x = batch
                x = x.to(self.device)
                t = sample_timesteps(x.size(0), timesteps, device=self.device)
                x_t, noise = q_sample(
                    x,
                    t,
                    self.model.sqrt_alphas_cumprod,
                    self.model.sqrt_one_minus_alphas_cumprod,
                )

                predicted_noise = self.model(x_t, t)
                loss = self.loss_fn(predicted_noise, noise)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                loop.set_postfix(loss=loss.item())
