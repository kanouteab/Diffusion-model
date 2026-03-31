"""UNet minimal pour les modèles de diffusion d'images."""
import math

import torch
import torch.nn as nn

from .scheduler import linear_beta_schedule


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor):
        device = timesteps.device
        half_dim = self.embedding_dim // 2
        exponent = -math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(timesteps.shape[0], 1, device=device)], dim=-1)
        return emb


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )
        self.time_mlp = None
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_ch),
            )

    def forward(self, x, t=None):
        h = self.conv(x)
        if t is not None and self.time_mlp is not None:
            time_emb = self.time_mlp(t).view(-1, h.shape[1], 1, 1)
            h = h + time_emb
        return h


class UNet(nn.Module):
    def __init__(self, img_channels=3, base_channel=64, timesteps=1000):
        super().__init__()
        self.timesteps = timesteps
        self.register_buffer("betas", linear_beta_schedule(timesteps))
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas", torch.sqrt(alphas))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

        time_dim = base_channel * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        # Encodeur
        self.inc = Block(img_channels, base_channel, time_dim)              # 3 -> 64
        self.down1 = Block(base_channel, base_channel * 2, time_dim)        # 64 -> 128
        self.down2 = Block(base_channel * 2, base_channel * 4, time_dim)    # 128 -> 256
        self.down3 = Block(base_channel * 4, base_channel * 8, time_dim)    # 256 -> 512
        self.bottleneck = Block(base_channel * 8, base_channel * 8, time_dim)  # 512 -> 512

        # Décodeur corrigé
        self.up1 = Block(base_channel * 8 + base_channel * 8, base_channel * 4, time_dim)  # 512+512 -> 256
        self.up2 = Block(base_channel * 4 + base_channel * 4, base_channel * 2, time_dim)  # 256+256 -> 128
        self.up3 = Block(base_channel * 2 + base_channel * 2, base_channel, time_dim)       # 128+128 -> 64
        self.up4 = Block(base_channel + base_channel, base_channel, time_dim)                # 64+64 -> 64

        self.out = nn.Conv2d(base_channel, img_channels, 1)

    def forward(self, x, t):
        t = self.time_mlp(t)

        # Encodeur
        e1 = self.inc(x, t)
        e2 = self.down1(nn.functional.avg_pool2d(e1, 2), t)
        e3 = self.down2(nn.functional.avg_pool2d(e2, 2), t)
        e4 = self.down3(nn.functional.avg_pool2d(e3, 2), t)
        b = self.bottleneck(nn.functional.avg_pool2d(e4, 2), t)

        # Décodeur
        d1 = nn.functional.interpolate(b, scale_factor=2, mode="nearest")
        d1 = torch.cat([d1, e4], dim=1)
        d1 = self.up1(d1, t)

        d2 = nn.functional.interpolate(d1, scale_factor=2, mode="nearest")
        d2 = torch.cat([d2, e3], dim=1)
        d2 = self.up2(d2, t)

        d3 = nn.functional.interpolate(d2, scale_factor=2, mode="nearest")
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.up3(d3, t)

        d4 = nn.functional.interpolate(d3, scale_factor=2, mode="nearest")
        d4 = torch.cat([d4, e1], dim=1)
        d4 = self.up4(d4, t)

        return self.out(d4)


class LegacyBlock(nn.Module):
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


class LegacyUNet(nn.Module):
    def __init__(self, img_channels=3, base_channel=64, timesteps=1000):
        super().__init__()
        self.timesteps = timesteps
        self.betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("sqrt_alphas", torch.sqrt(alphas))
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

        self.enc1 = LegacyBlock(img_channels, base_channel)
        self.enc2 = LegacyBlock(base_channel, base_channel * 2)
        self.dec1 = LegacyBlock(base_channel * 2 + base_channel, base_channel)
        self.out = nn.Conv2d(base_channel, img_channels, 1)

    def forward(self, x, t):
        e1 = self.enc1(x)
        e2 = self.enc2(nn.functional.avg_pool2d(e1, 2))
        d1 = nn.functional.interpolate(e2, scale_factor=2, mode="nearest")
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.out(d1)