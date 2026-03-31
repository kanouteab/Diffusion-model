"""Compare plusieurs checkpoints de diffusion en générant une grille d'images côte à côte."""
import argparse
import os

import torch
from PIL import Image
from torchvision.utils import make_grid, save_image

from diffusion_model import LegacyUNet, UNet
from diffusion_model.noise import p_sample_loop


def load_model(checkpoint, device, timesteps):
    model = UNet(img_channels=3, base_channel=64, timesteps=timesteps).to(device)
    state = torch.load(checkpoint, map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        print(f"Checkpoint incompatible avec UNet. Chargement du LegacyUNet pour {checkpoint}.")
        model = LegacyUNet(img_channels=3, base_channel=64, timesteps=timesteps).to(device)
        model.load_state_dict(state, strict=False)
    model.eval()
    return model


def concat_grids(grids):
    if not grids:
        raise ValueError("Aucune grille à concaténer")
    widths = [grid.shape[2] for grid in grids]
    height = grids[0].shape[1]
    concat = torch.cat(grids, dim=2)
    return concat


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    os.makedirs(os.path.dirname(args.output_image) or ".", exist_ok=True)
    generated_grids = []

    for checkpoint in args.checkpoints:
        model = load_model(checkpoint, device, args.timesteps)
        torch.manual_seed(args.seed)
        with torch.no_grad():
            samples = p_sample_loop(
                model,
                (args.num_samples, 3, args.image_size, args.image_size),
                args.timesteps,
                device,
            )
        samples = (samples.clamp(-1, 1) + 1) / 2.0
        generated_grids.append(make_grid(samples, nrow=args.num_samples))

    comparison_grid = concat_grids(generated_grids)
    save_image(comparison_grid, args.output_image)
    print(f"Comparaison sauvegardée: {args.output_image}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare plusieurs checkpoints par génération d'images")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True)
    parser.add_argument("--output-image", type=str, default="checkpoint_comparison.png")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    main(args)
