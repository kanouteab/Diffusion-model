"""Propagation de bruit pour DDPM (débruitage)."""
import torch


def sample_timesteps(batch_size: int, timesteps: int, device=None):
    return torch.randint(0, timesteps, (batch_size,), device=device)


def q_sample(
    x_start: torch.Tensor,
    t: torch.Tensor,
    sqrt_alphas_cumprod: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
):
    """Échantillonne x_t à partir de x_0 avec un bruit gaussien."""
    noise = torch.randn_like(x_start)
    return (
        sqrt_alphas_cumprod[t].view(-1, 1, 1, 1) * x_start
        + sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1) * noise,
        noise,
    )


def p_sample_loop(model, shape, timesteps, device, return_intermediates=False, frame_interval=10):
    """Échantillonne depuis bruit pur vers image reconstruite."""
    b = shape[0]
    img = torch.randn(shape, device=device)
    frames = []

    for i in reversed(range(timesteps)):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        predicted_noise = model(img, t)

        beta = model.betas[i]
        sqrt_one_minus_alphas_cumprod = model.sqrt_one_minus_alphas_cumprod[i]
        sqrt_recip_alphas = 1.0 / model.sqrt_alphas[i]

        if i > 0:
            posterior_variance = beta * (
                1.0 - model.alphas_cumprod[i - 1]
            ) / (1.0 - model.alphas_cumprod[i])
            noise = torch.randn_like(img)
        else:
            posterior_variance = torch.tensor(0.0, device=device, dtype=img.dtype)
            noise = torch.zeros_like(img)

        mean = sqrt_recip_alphas * (
            img - beta / sqrt_one_minus_alphas_cumprod * predicted_noise
        )
        img = mean + torch.sqrt(posterior_variance) * noise

        if return_intermediates and (i % frame_interval == 0 or i == 0):
            frames.append(img.detach().cpu())

    return (img, frames) if return_intermediates else img
