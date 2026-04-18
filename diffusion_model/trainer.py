import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from .noise import q_sample, sample_timesteps


class Trainer:
    def __init__(self, model, dataloader, lr=2e-4, device="cpu", weight_decay=0.0, l1_weight=0.05):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.optim = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optim, mode="min", factor=0.5, patience=2)
        self.loss_fn = nn.MSELoss()
        self.l1_fn = nn.L1Loss()
        self.l1_weight = l1_weight

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
        best_model_path=None,
        grad_clip=None,
    ):
        timesteps = timesteps or self.model.timesteps
        history = {"train_loss": [], "val_loss": [], "lr": []}
        best_val_loss = float("inf")

        for epoch in range(epochs):
            self.model.train()
            loop = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            epoch_loss = 0.0
            epoch_samples = 0

            for batch in loop:
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                x = x.to(self.device)

                t = sample_timesteps(x.size(0), timesteps, device=self.device)
                x_t, noise = q_sample(
                    x,
                    t,
                    self.model.sqrt_alphas_cumprod,
                    self.model.sqrt_one_minus_alphas_cumprod,
                )

                pred = self.model(x_t, t)

                mse = self.loss_fn(pred, noise)
                l1 = self.l1_fn(pred, noise)
                loss = mse + self.l1_weight * l1

                self.optim.zero_grad()
                loss.backward()

                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

                self.optim.step()

                loop.set_postfix(loss=loss.item())
                epoch_loss += loss.item() * x.size(0)
                epoch_samples += x.size(0)

            avg_train = epoch_loss / epoch_samples
            history["train_loss"].append(avg_train)

            lr = self.optim.param_groups[0]["lr"]
            history["lr"].append(lr)

            print(f"Epoch {epoch+1}: train_loss={avg_train:.6f} | lr={lr:.6e}")

            if val_dataloader:
                self.model.eval()
                val_loss = 0.0
                val_samples = 0

                with torch.no_grad():
                    for idx, batch in enumerate(val_dataloader):
                        x = batch[0] if isinstance(batch, (tuple, list)) else batch
                        x = x.to(self.device)

                        t = sample_timesteps(x.size(0), timesteps, device=self.device)
                        x_t, noise = q_sample(
                            x,
                            t,
                            self.model.sqrt_alphas_cumprod,
                            self.model.sqrt_one_minus_alphas_cumprod,
                        )

                        pred = self.model(x_t, t)
                        loss = self.loss_fn(pred, noise) + self.l1_weight * self.l1_fn(pred, noise)

                        val_loss += loss.item() * x.size(0)
                        val_samples += x.size(0)

                        if val_num_batches and idx + 1 >= val_num_batches:
                            break

                avg_val = val_loss / val_samples
                history["val_loss"].append(avg_val)

                print(f"Epoch {epoch+1}: val_loss={avg_val:.6f}")

                self.scheduler.step(avg_val)

                if best_model_path and avg_val < best_val_loss:
                    best_val_loss = avg_val
                    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                    torch.save(self.model.state_dict(), best_model_path)
                    print(f"Best model sauvegardé: {best_model_path}")

            if checkpoint_interval and checkpoint_dir and (epoch + 1) % checkpoint_interval == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
                torch.save(self.model.state_dict(), path)

            if sample_interval and sample_fn and (epoch + 1) % sample_interval == 0:
                sample_fn(epoch + 1)

        return history
