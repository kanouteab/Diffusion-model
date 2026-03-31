"""Évaluation simple d'un modèle de diffusion sur CIFAR-10."""
import argparse

import torch
import torch.nn as nn
from diffusion_model import LegacyUNet, UNet
from diffusion_model.noise import q_sample, sample_timesteps
from diffusion_model.utils import get_dataloader


def load_model(checkpoint, device, timesteps):
    model = UNet(img_channels=3, base_channel=64, timesteps=timesteps).to(device)
    state = torch.load(checkpoint, map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        print(f"Checkpoint incompatible avec UNet. Chargement du LegacyUNet pour {checkpoint}.")
        model = LegacyUNet(img_channels=3, base_channel=64, timesteps=timesteps).to(device)
        model.load_state_dict(state, strict=False)
    return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_model(args.model, device, args.timesteps)
    model.eval()

    dataloader = get_dataloader(
        batch_size=args.batch_size,
        image_size=args.image_size,
        train=False,
        num_workers=args.num_workers,
        subset_size=args.subset_size,
    )

    criterion = nn.MSELoss(reduction="mean")
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if isinstance(batch, (tuple, list)):
                x, _ = batch
            else:
                x = batch
            x = x.to(device)
            t = sample_timesteps(x.size(0), args.timesteps, device=device)
            x_t, noise = q_sample(
                x,
                t,
                model.sqrt_alphas_cumprod,
                model.sqrt_one_minus_alphas_cumprod,
            )
            predicted_noise = model(x_t, t)
            loss = criterion(predicted_noise, noise)
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
            if args.num_batches and idx + 1 >= args.num_batches:
                break

    avg_loss = total_loss / max(1, total_samples)
    print(f"Évaluation terminée. MSE moyen sur {total_samples} images: {avg_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évalue un modèle de diffusion sur CIFAR-10")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--subset-size", type=int, default=1000, help="Taille du sous-ensemble de validation")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-batches", type=int, default=10, help="Nombre de batches de validation à utiliser")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    main(args)
