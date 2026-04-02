"""Entraînement du modèle de diffusion."""
import copy
import os

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from diffusion_model.noise import q_sample, sample_timesteps


class EMA:
    def __init__(self, beta=0.999):
        self.beta = beta

    def update_model_average(self, ema_model, current_model):
        for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
            ema_params.data = self.beta * ema_params.data + (1.0 - self.beta) * current_params.data

    def update_buffers(self, ema_model, current_model):
        for current_buffer, ema_buffer in zip(current_model.buffers(), ema_model.buffers()):
            ema_buffer.data.copy_(current_buffer.data)


class Trainer:
    def __init__(
        self,
        model,
        dataloader,
        val_dataloader=None,
        lr=2e-4,
        device="cpu",
        use_ema=True,
        ema_beta=0.999,
    ):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.optim = Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.scheduler = None

        self.use_ema = use_ema
        self.ema_beta = ema_beta
        self.ema_helper = EMA(beta=ema_beta)

        if self.use_ema:
            self.ema_model = copy.deepcopy(self.model).to(device)
            self.ema_model.eval()
            for param in self.ema_model.parameters():
                param.requires_grad = False
        else:
            self.ema_model = None

    def _build_scheduler(self, epochs):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, T_max=epochs
        )

    def _run_validation(self, timesteps):
        if self.val_dataloader is None:
            return None

        eval_model = self.ema_model if self.use_ema and self.ema_model is not None else self.model
        eval_model.eval()

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
                    eval_model.sqrt_alphas_cumprod,
                    eval_model.sqrt_one_minus_alphas_cumprod,
                )

                predicted_noise = eval_model(x_t, t)
                loss = self.loss_fn(predicted_noise, noise)
                val_loss += loss.item()

        val_loss /= len(self.val_dataloader)
        return val_loss

    def _save_best_model(self, path="best_model.pth"):
        save_model = self.ema_model if self.use_ema and self.ema_model is not None else self.model
        torch.save(
            {
                "model_state_dict": save_model.state_dict(),
                "base_channel": save_model.time_mlp[1].in_features // 2,
                "timesteps": save_model.timesteps,
                "use_ema": self.use_ema,
            },
            path,
        )

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

        self._build_scheduler(epochs)

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

                if self.use_ema and self.ema_model is not None:
                    self.ema_helper.update_model_average(self.ema_model, self.model)
                    self.ema_helper.update_buffers(self.ema_model, self.model)

                train_loss += loss.item()
                loop.set_postfix(loss=loss.item(), lr=self.optim.param_groups[0]["lr"])

            train_loss /= len(self.dataloader)
            print(f"Loss moyenne train époque {epoch+1}/{epochs}: {train_loss:.4f}")

            val_loss = self._run_validation(timesteps)
            if val_loss is not None:
                print(f"Loss moyenne validation époque {epoch+1}/{epochs}: {val_loss:.4f}")

            current_loss = val_loss if val_loss is not None else train_loss

            if current_loss < best_loss:
                best_loss = current_loss
                self._save_best_model("best_model.pth")
                metric_name = "val_loss" if val_loss is not None else "train_loss"
                print(f"✅ Nouveau meilleur modèle sauvegardé : best_model.pth ({metric_name}={best_loss:.4f})")

            if self.scheduler is not None:
                self.scheduler.step()

            epoch_num = epoch + 1

            if checkpoint_interval and checkpoint_dir and epoch_num % checkpoint_interval == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch_num}.pth")

                save_model = self.ema_model if self.use_ema and self.ema_model is not None else self.model
                torch.save(
                    {
                        "model_state_dict": save_model.state_dict(),
                        "base_channel": save_model.time_mlp[1].in_features // 2,
                        "timesteps": save_model.timesteps,
                        "use_ema": self.use_ema,
                    },
                    checkpoint_path,
                )
                print(f"Checkpoint sauvegardé: {checkpoint_path}")

            if sample_interval and sample_fn and epoch_num % sample_interval == 0:
                sample_model = self.ema_model if self.use_ema and self.ema_model is not None else self.model
                sample_fn(epoch_num, sample_model)