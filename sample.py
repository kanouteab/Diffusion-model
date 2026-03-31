"""Script de génération d'images à partir d'un modèle de diffusion entraîné."""
import argparse
import os

import torch
from PIL import Image
from torchvision.utils import make_grid, save_image

from diffusion_model import LegacyUNet, UNet
from diffusion_model.noise import p_sample_loop


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = UNet(img_channels=3, base_channel=64, timesteps=args.timesteps).to(device)
    try:
        model.load_state_dict(torch.load(args.model, map_location=device))
    except RuntimeError:
        print("Checkpoint incompatible avec le UNet actuel : chargement du LegacyUNet.")
        model = LegacyUNet(img_channels=3, base_channel=64, timesteps=args.timesteps).to(device)
        model.load_state_dict(torch.load(args.model, map_location=device), strict=False)
    model.eval()

    with torch.no_grad():
        if args.output_gif:
            out, frames = p_sample_loop(
                model,
                (args.num_samples, 3, args.image_size, args.image_size),
                args.timesteps,
                device,
                return_intermediates=True,
                frame_interval=args.gif_frame_interval,
            )
        else:
            out = p_sample_loop(
                model,
                (args.num_samples, 3, args.image_size, args.image_size),
                args.timesteps,
                device,
            )
    out = (out.clamp(-1, 1) + 1) / 2.0

    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    torch.save(out, args.output)

    image_path = args.output_image or os.path.splitext(args.output)[0] + ".png"
    save_image(out, image_path, nrow=min(8, args.num_samples))

    print(f"Échantillons générés: {args.output}")
    print(f"Image sauvegardée: {image_path}")

    if args.output_gif:
        gif_path = args.output_gif
        os.makedirs(os.path.dirname(gif_path) or ".", exist_ok=True)
        grid_frames = []
        for frame in frames:
            grid = make_grid((frame.clamp(-1, 1) + 1) / 2.0, nrow=min(8, args.num_samples))
            ndarr = (grid.mul(255).permute(1, 2, 0).byte().cpu().numpy())
            grid_frames.append(Image.fromarray(ndarr))
        grid_frames[0].save(
            gif_path,
            save_all=True,
            append_images=grid_frames[1:],
            duration=int(1000 / max(1, args.gif_fps)),
            loop=0,
        )
        print(f"GIF sauvegardé: {gif_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génère des images par diffusion")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, default="samples.pt")
    parser.add_argument("--output-image", type=str, default=None)
    parser.add_argument("--output-gif", type=str, default=None)
    parser.add_argument("--gif-frame-interval", type=int, default=10)
    parser.add_argument("--gif-fps", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    main(args)
