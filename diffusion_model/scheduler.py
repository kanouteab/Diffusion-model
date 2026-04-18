"""Schedule de bruit pour le modèle de diffusion."""
import torch


def linear_beta_schedule(timesteps: int, beta_start=1e-4, beta_end=0.02) -> torch.Tensor:
    """Retourne un vecteur beta linéaire entre beta_start et beta_end."""
    return torch.linspace(beta_start, beta_end, timesteps)
