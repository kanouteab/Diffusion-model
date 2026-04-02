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
        val_dataloader=None,
        val_num_batches=10,
    ):
        timesteps = timesteps or self.model.timesteps
        self.model.train()
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            loop = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            epoch_loss = 0.0
            epoch_samples = 0

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
                epoch_loss += loss.item() * x.size(0)
                epoch_samples += x.size(0)

            avg_train_loss = epoch_loss / max(1, epoch_samples)
            history["train_loss"].append(avg_train_loss)
            print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.6f}")

            if val_dataloader is not None:
                self.model.eval()
                val_loss = 0.0
                val_samples = 0
                with torch.no_grad():
                    for idx, val_batch in enumerate(val_dataloader):
                        if isinstance(val_batch, (tuple, list)):
                            x_val, _ = val_batch
                        else:
                            x_val = val_batch

                        x_val = x_val.to(self.device)
                        t_val = sample_timesteps(x_val.size(0), timesteps, device=self.device)
                        x_t_val, noise_val = q_sample(
                            x_val,
                            t_val,
                            self.model.sqrt_alphas_cumprod,
                            self.model.sqrt_one_minus_alphas_cumprod,
                        )
                        predicted_noise_val = self.model(x_t_val, t_val)
                        batch_val_loss = self.loss_fn(predicted_noise_val, noise_val)
                        val_loss += batch_val_loss.item() * x_val.size(0)
                        val_samples += x_val.size(0)

                        if val_num_batches and idx + 1 >= val_num_batches:
                            break

                avg_val_loss = val_loss / max(1, val_samples)
                history["val_loss"].append(avg_val_loss)
                print(f"Epoch {epoch+1}: val_loss={avg_val_loss:.6f}")
                self.model.train()

            epoch_num = epoch + 1
            if checkpoint_interval and checkpoint_dir and epoch_num % checkpoint_interval == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch_num}.pth")
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Checkpoint sauvegardé: {checkpoint_path}")

            if sample_interval and sample_fn and epoch_num % sample_interval == 0:
                sample_fn(epoch_num)

        return history
