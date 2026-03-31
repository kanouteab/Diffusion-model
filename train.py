import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / max(half_dim - 1, 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int | None = None):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

        if time_emb_dim is not None:
            self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        else:
            self.time_mlp = None

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor | None = None) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))

        if self.time_mlp is not None and t_emb is not None:
            time_feature = self.time_mlp(t_emb)
            time_feature = time_feature[:, :, None, None]
            x = x + time_feature

        x = self.act(self.bn2(self.conv2(x)))
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, time_emb_dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x, t_emb)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels, time_emb_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Ajustement spatial si jamais il y a une légère différence
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x, t_emb)
        return x


class UNet(nn.Module):
    def __init__(self, img_channels: int = 3, base_channel: int = 64, timesteps: int = 1000):
        super().__init__()

        time_emb_dim = base_channel * 4

        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channel),
            nn.Linear(base_channel, time_emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Encoder
        self.enc1 = DoubleConv(img_channels, base_channel, time_emb_dim)
        self.enc2 = DownBlock(base_channel, base_channel * 2, time_emb_dim)
        self.enc3 = DownBlock(base_channel * 2, base_channel * 4, time_emb_dim)

        # Bottleneck
        self.bottleneck = DownBlock(base_channel * 4, base_channel * 8, time_emb_dim)

        # Decoder
        self.up1 = UpBlock(base_channel * 8, base_channel * 4, base_channel * 4, time_emb_dim)
        self.up2 = UpBlock(base_channel * 4, base_channel * 2, base_channel * 2, time_emb_dim)
        self.up3 = UpBlock(base_channel * 2, base_channel, base_channel, time_emb_dim)

        # Sortie
        self.out_conv = nn.Conv2d(base_channel, img_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embedding(t)

        # Encoder
        e1 = self.enc1(x, t_emb)   # [B, 64, H, W]
        e2 = self.enc2(e1, t_emb)  # [B, 128, H/2, W/2]
        e3 = self.enc3(e2, t_emb)  # [B, 256, H/4, W/4]

        # Bottleneck
        b = self.bottleneck(e3, t_emb)  # [B, 512, H/8, W/8]

        # Decoder
        d1 = self.up1(b, e3, t_emb)   # [B, 256, H/4, W/4]
        d2 = self.up2(d1, e2, t_emb)  # [B, 128, H/2, W/2]
        d3 = self.up3(d2, e1, t_emb)  # [B, 64, H, W]

        out = self.out_conv(d3)
        return out
