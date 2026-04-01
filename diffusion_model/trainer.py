"""Entraînement du modèle de diffusion."""
import os

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

    def train(
        self,
        epochs=10,
        timesteps=None,
        checkpoint_interval=None,
        checkpoint_dir=None,
        sample_interval=None,
        sample_fn=None,
    ):
        timesteps = timesteps or self.model.timesteps
        self.model.train()

        # Meilleure loss observée jusque-là
        best_loss = float("inf")

        for epoch in range(epochs):
            loop = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            epoch_loss = 0.0

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

                epoch_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            # Loss moyenne de l'époque
            epoch_loss /= len(self.dataloader)
            print(f"Loss moyenne époque {epoch+1}/{epochs}: {epoch_loss:.4f}")

            # Sauvegarde du meilleur modèle
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(self.model.state_dict(), "best_model.pth")
                print(f"✅ Nouveau meilleur modèle sauvegardé : best_model.pth (loss={best_loss:.4f})")

            epoch_num = epoch + 1

            # Sauvegarde périodique des checkpoints
            if checkpoint_interval and checkpoint_dir and epoch_num % checkpoint_interval == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"checkpoint_epoch_{epoch_num}.pth"
                )
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Checkpoint sauvegardé: {checkpoint_path}")

            # Génération périodique d'échantillons
            if sample_interval and sample_fn and epoch_num % sample_interval == 0:
                sample_fn(epoch_num)