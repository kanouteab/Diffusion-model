"""UNet minimal pour les modèles de diffusion d'images."""
import math

import torch
import torch.nn as nn

from .scheduler import linear_beta_schedule


class SinusoidalPositionEmbeddings(nn.Module):
    # Encode les pas de temps t en vecteurs continus (sin/cos), comme dans les Transformers.
    def __init__(self, embedding_dim):
        super().__init__()
        # Dimension finale du vecteur d'embedding temporel.
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor):
        # Les calculs sont effectués sur le même device que t.
        device = timesteps.device
        # Moitié de la dimension pour sin, moitié pour cos.
        half_dim = self.embedding_dim // 2
        # Protection numérique: évite une division par zéro si half_dim == 1.
        denom = max(half_dim - 1, 1)
        exponent = -math.log(10000.0) / denom
        # Construction des fréquences géométriquement espacées.
        emb = torch.exp(torch.arange(half_dim, device=device).float() * exponent)
        # Produit t * fréquence pour chaque composante.
        emb = timesteps[:, None].float() * emb[None, :]
        # Concaténation sin/cos -> représentation périodique de t.
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        # Si embedding_dim est impair, on pad avec un zéro pour garder la taille demandée.
        if self.embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(timesteps.shape[0], 1, device=device)], dim=-1)
        return emb


class Block(nn.Module):
    # Bloc conv standard du UNet, avec conditionnement temporel optionnel.
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        # Double convolution + normalisation + activation SiLU.
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )
        # Projection temporelle ajoutée au tenseur de features si fournie.
        self.time_mlp = None
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_ch),
            )

    def forward(self, x, t=None):
        # Chemin convolutif principal.
        h = self.conv(x)
        # Conditionnement additif par embedding temporel (broadcast spatial HxW).
        if t is not None and self.time_mlp is not None:
            time_emb = self.time_mlp(t).view(-1, h.shape[1], 1, 1)
            h = h + time_emb
        return h


class UNet(nn.Module):
    # UNet conditionné par le temps pour prédire le bruit à chaque étape de diffusion.
    def __init__(self, img_channels=3, base_channel=64, timesteps=1000):
        super().__init__()
        # Nombre total d'étapes de diffusion supportées par ce modèle.
        self.timesteps = timesteps
        # Buffers du schedule (déplacés automatiquement CPU/GPU avec le modèle).
        self.register_buffer("betas", linear_beta_schedule(timesteps))
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas", torch.sqrt(alphas))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

        # Dimension du vecteur temporel utilisé pour conditionner les blocs.
        time_dim = base_channel * 4
        # Petit MLP temporel: sinusoidal embedding -> projection non linéaire.
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        # Encodeur (descente de résolution).
        self.inc = Block(img_channels, base_channel, time_dim)
        self.down1 = Block(base_channel, base_channel * 2, time_dim)
        self.down2 = Block(base_channel * 2, base_channel * 4, time_dim)
        self.down3 = Block(base_channel * 4, base_channel * 8, time_dim)
        self.bottleneck = Block(base_channel * 8, base_channel * 8, time_dim)

        # Décodeur (remontée de résolution) avec skip connections concaténées.
        # up1: b(8c) upsample + e4(8c) -> 16c en entrée.
        self.up1 = Block(base_channel * 16, base_channel * 4, time_dim)
        # up2: up1(4c) upsample + e3(4c) -> 8c en entrée.
        self.up2 = Block(base_channel * 8, base_channel * 2, time_dim)
        # up3: up2(2c) upsample + e2(2c) -> 4c en entrée.
        self.up3 = Block(base_channel * 4, base_channel, time_dim)
        # Projection finale vers le nombre de canaux image (bruit prédit).
        # d4 concatène up3(c) et e1(c), donc entrée à 2c.
        self.out = nn.Conv2d(base_channel * 2, img_channels, 1)

    def forward(self, x, t):
        # Encodage temporel partagé par tous les blocs conditionnés.
        t = self.time_mlp(t)

        # Encodeur: extraction hiérarchique des features.
        e1 = self.inc(x, t)
        e2 = self.down1(nn.functional.avg_pool2d(e1, 2), t)
        e3 = self.down2(nn.functional.avg_pool2d(e2, 2), t)
        e4 = self.down3(nn.functional.avg_pool2d(e3, 2), t)
        b = self.bottleneck(nn.functional.avg_pool2d(e4, 2), t)

        # Décodeur: upsample + concat skip + raffinement.
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
        # Sortie: bruit prédit de même taille que l'entrée.
        return self.out(d4)


class LegacyBlock(nn.Module):
    # Ancien bloc conv simple (sans conditionnement temporel explicite).
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
    # Version historique du UNet conservée pour rétro-compatibilité checkpoints.
    def __init__(self, img_channels=3, base_channel=64, timesteps=1000):
        super().__init__()
        # Timesteps associés au checkpoint Legacy.
        self.timesteps = timesteps
        # Buffers du schedule Legacy (doivent suivre le device du modèle).
        self.register_buffer("betas", torch.linspace(1e-4, 0.02, timesteps))
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("sqrt_alphas", torch.sqrt(alphas))
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

        # Architecture Legacy plus compacte.
        self.enc1 = LegacyBlock(img_channels, base_channel)
        self.enc2 = LegacyBlock(base_channel, base_channel * 2)
        self.dec1 = LegacyBlock(base_channel * 2 + base_channel, base_channel)
        self.out = nn.Conv2d(base_channel, img_channels, 1)

    def forward(self, x, t):
        # t est conservé dans la signature pour compatibilité API avec UNet.
        e1 = self.enc1(x)
        e2 = self.enc2(nn.functional.avg_pool2d(e1, 2))
        d1 = nn.functional.interpolate(e2, scale_factor=2, mode="nearest")
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.out(d1)
