"""Propagation de bruit pour DDPM (débruitage)."""
import torch


def sample_timesteps(batch_size: int, timesteps: int, device=None):
    return torch.randint(0, timesteps, (batch_size,), device=device)


def q_sample(x_start: torch.Tensor, t: torch.Tensor, sqrt_alphas_cumprod: torch.Tensor, sqrt_one_minus_alphas_cumprod: torch.Tensor):
    """Échantillonne x_t à partir de x_0 avec un bruit gaussien."""
    noise = torch.randn_like(x_start)
    return sqrt_alphas_cumprod[t].view(-1, 1, 1, 1) * x_start + sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1) * noise, noise


def p_sample_loop(model, shape, timesteps, device):
    """Échantillonne depuis bruit pur vers image reconstruite."""
    b = shape[0]
    img = torch.randn(shape, device=device)

    for i in reversed(range(timesteps)):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        predicted_noise = model(img, t)

        if i > 0:
            noise = torch.randn_like(img)
        else:
            noise = torch.zeros_like(img)

        beta = model.betas[i]
        sqrt_one_minus_alphas_cumprod = model.sqrt_one_minus_alphas_cumprod[i]
        sqrt_recip_alphas = 1.0 / model.sqrt_alphas[i]

        img = sqrt_recip_alphas * (img - beta / sqrt_one_minus_alphas_cumprod * predicted_noise) + torch.sqrt(beta) * noise

    return img
