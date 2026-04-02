"""Entraînement du modèle de diffusion."""
import os

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from diffusion_model.noise import q_sample, sample_timesteps


class Trainer:
    def __init__(self, model, dataloader, val_dataloader=None, lr=2e-4, device="cpu"):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
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
        best_loss = float("inf")

        for epoch in range(epochs):
            self.model.train()
            loop = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            train_loss = 0.0

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

                train_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            train_loss /= len(self.dataloader)
            print(f"Loss moyenne train époque {epoch+1}/{epochs}: {train_loss:.4f}")

            val_loss = None
            if self.val_dataloader is not None:
                self.model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch in self.val_dataloader:
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
                        val_loss += loss.item()

                val_loss /= len(self.val_dataloader)
                print(f"Loss moyenne validation époque {epoch+1}/{epochs}: {val_loss:.4f}")

            current_loss = val_loss if val_loss is not None else train_loss

            if current_loss < best_loss:
                best_loss = current_loss
                torch.save(self.model.state_dict(), "best_model.pth")
                metric_name = "val_loss" if val_loss is not None else "train_loss"
                print(
                    f"✅ Nouveau meilleur modèle sauvegardé : best_model.pth "
                    f"({metric_name}={best_loss:.4f})"
                )

            epoch_num = epoch + 1

            if checkpoint_interval and checkpoint_dir and epoch_num % checkpoint_interval == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"checkpoint_epoch_{epoch_num}.pth"
                )
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Checkpoint sauvegardé: {checkpoint_path}")

            if sample_interval and sample_fn and epoch_num % sample_interval == 0:
                sample_fn(epoch_num)