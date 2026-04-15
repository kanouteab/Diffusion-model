"""Évaluation de débruitage image-à-image pour modèle de diffusion."""
import argparse
import json
import math
import os

import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid

from diffusion_model import load_model_from_checkpoint
from diffusion_model.noise import q_sample
from diffusion_model.utils import get_dataloader


def tensor_to_01(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1.0) / 2.0


def compute_mse(x: torch.Tensor, y: torch.Tensor) -> float:
    return F.mse_loss(x, y).item()


def compute_psnr(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0) -> float:
    mse = F.mse_loss(x, y).item()
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)


def compute_ssim_simple(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Approximation simple de SSIM globale sur batch.
    x, y attendus dans [0,1].
    """
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_x = x.mean(dim=(1, 2, 3), keepdim=True)
    mu_y = y.mean(dim=(1, 2, 3), keepdim=True)

    sigma_x = ((x - mu_x) ** 2).mean(dim=(1, 2, 3), keepdim=True)
    sigma_y = ((y - mu_y) ** 2).mean(dim=(1, 2, 3), keepdim=True)
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean(dim=(1, 2, 3), keepdim=True)

    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    )
    return ssim.mean().item()


def denoise_one_step(model, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Reconstruction approchée de x0 à partir de x_t et du bruit prédit.
    """
    predicted_noise = model(x_t, t)

    sqrt_alpha_bar = model.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar = model.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

    x0_pred = (x_t - sqrt_one_minus_alpha_bar * predicted_noise) / sqrt_alpha_bar
    return x0_pred.clamp(-1, 1)


def save_comparison_grid(clean, noisy, restored, output_path: str, max_items: int = 8):
    clean = tensor_to_01(clean[:max_items])
    noisy = tensor_to_01(noisy[:max_items])
    restored = tensor_to_01(restored[:max_items])

    rows = []
    for i in range(clean.size(0)):
        rows.append(torch.stack([clean[i], noisy[i], restored[i]], dim=0))

    grid_batch = torch.cat(rows, dim=0)
    grid = make_grid(grid_batch, nrow=3)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_image(grid, output_path)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Checkpoint introuvable: {args.model}")

    model = load_model_from_checkpoint(args.model, device, args.timesteps)
    model.eval()

    dataloader = get_dataloader(
        batch_size=args.batch_size,
        image_size=args.image_size,
        train=False,
        num_workers=args.num_workers,
        subset_size=args.subset_size,
    )

    timestep = min(args.noise_timestep, model.timesteps - 1)

    total_mse_noisy = 0.0
    total_mse_restored = 0.0
    total_psnr_noisy = 0.0
    total_psnr_restored = 0.0
    total_ssim_noisy = 0.0
    total_ssim_restored = 0.0
    total_samples = 0

    first_clean = None
    first_noisy = None
    first_restored = None

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            x, _ = batch if isinstance(batch, (tuple, list)) else (batch, None)
            x = x.to(device)
            bsz = x.size(0)

            t = torch.full((bsz,), timestep, device=device, dtype=torch.long)
            x_t, _ = q_sample(
                x,
                t,
                model.sqrt_alphas_cumprod,
                model.sqrt_one_minus_alphas_cumprod,
            )
            x_restored = denoise_one_step(model, x_t, t)

            x_01 = tensor_to_01(x)
            x_t_01 = tensor_to_01(x_t)
            x_restored_01 = tensor_to_01(x_restored)

            total_mse_noisy += F.mse_loss(x_t_01, x_01, reduction="sum").item()
            total_mse_restored += F.mse_loss(x_restored_01, x_01, reduction="sum").item()

            total_psnr_noisy += compute_psnr(x_t_01, x_01) * bsz
            total_psnr_restored += compute_psnr(x_restored_01, x_01) * bsz

            total_ssim_noisy += compute_ssim_simple(x_t_01, x_01) * bsz
            total_ssim_restored += compute_ssim_simple(x_restored_01, x_01) * bsz

            total_samples += bsz

            if first_clean is None:
                first_clean = x.detach().cpu()
                first_noisy = x_t.detach().cpu()
                first_restored = x_restored.detach().cpu()

            if args.num_batches and idx + 1 >= args.num_batches:
                break

    metrics = {
        "model": args.model,
        "noise_timestep": timestep,
        "num_samples": total_samples,
        "mse_noisy": total_mse_noisy / max(1, total_samples),
        "mse_restored": total_mse_restored / max(1, total_samples),
        "psnr_noisy": total_psnr_noisy / max(1, total_samples),
        "psnr_restored": total_psnr_restored / max(1, total_samples),
        "ssim_noisy": total_ssim_noisy / max(1, total_samples),
        "ssim_restored": total_ssim_restored / max(1, total_samples),
    }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if first_clean is not None:
        save_comparison_grid(first_clean, first_noisy, first_restored, args.output_image)

    print("Évaluation débruitage terminée.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation débruitage pour modèle de diffusion")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--noise-timestep", type=int, default=300)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--subset-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-batches", type=int, default=5)
    parser.add_argument("--output-json", type=str, default="outputs/denoising_metrics.json")
    parser.add_argument("--output-image", type=str, default="outputs/noisy_vs_restored_grid.png")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    main(args)
