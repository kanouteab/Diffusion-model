"""UNet minimal pour les modèles de diffusion d'images."""
import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, img_channels=3, base_channel=64, timesteps=1000):
        super().__init__()
        self.timesteps = timesteps
        self.betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("sqrt_alphas", torch.sqrt(alphas))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

        self.enc1 = Block(img_channels, base_channel)
        self.enc2 = Block(base_channel, base_channel * 2)
        # garde la concaténation d'encodeur pour le skip connection
        self.dec1 = Block(base_channel * 2 + base_channel, base_channel)
        self.out = nn.Conv2d(base_channel, img_channels, 1)

    def forward(self, x, t):
        # t est ignoré pour ce modèle minimal. Ajouter embedding temporel pour la qualité.
        e1 = self.enc1(x)
        e2 = self.enc2(nn.functional.avg_pool2d(e1, 2))
        d1 = nn.functional.interpolate(e2, scale_factor=2, mode="nearest")
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.out(d1)
