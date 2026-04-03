"""Génération d'exemples de débruitage."""
import argparse
import os

import torch
from torchvision.utils import save_image

from diffusion_model import load_model_from_checkpoint
from diffusion_model.noise import q_sample
from diffusion_model.utils import get_dataloader


def tensor_to_01(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1.0) / 2.0


def denoise_one_step(model, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    predicted_noise = model(x_t, t)
    sqrt_alpha_bar = model.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar = model.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    x0_pred = (x_t - sqrt_one_minus_alpha_bar * predicted_noise) / sqrt_alpha_bar
    return x0_pred.clamp(-1, 1)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Checkpoint introuvable: {args.model}")

    model = load_model_from_checkpoint(args.model, device, args.timesteps)
    model.eval()

    dataloader = get_dataloader(
        batch_size=args.num_samples,
        image_size=args.image_size,
        train=False,
        num_workers=args.num_workers,
        subset_size=args.num_samples,
    )

    batch = next(iter(dataloader))
    x, _ = batch if isinstance(batch, (tuple, list)) else (batch, None)
    x = x.to(device)

    timestep = min(args.noise_timestep, model.timesteps - 1)
    t = torch.full((x.size(0),), timestep, device=device, dtype=torch.long)

    with torch.no_grad():
        x_t, _ = q_sample(
            x,
            t,
            model.sqrt_alphas_cumprod,
            model.sqrt_one_minus_alphas_cumprod,
        )
        x_restored = denoise_one_step(model, x_t, t)

    os.makedirs(args.output_dir, exist_ok=True)

    save_image(tensor_to_01(x), os.path.join(args.output_dir, "clean.png"), nrow=min(8, args.num_samples))
    save_image(tensor_to_01(x_t), os.path.join(args.output_dir, "noisy.png"), nrow=min(8, args.num_samples))
    save_image(tensor_to_01(x_restored), os.path.join(args.output_dir, "restored.png"), nrow=min(8, args.num_samples))

    print(f"Exemples sauvegardés dans : {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exemples de débruitage")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--noise-timestep", type=int, default=300)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="outputs/denoising_examples")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    main(args)
