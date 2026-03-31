"""Script de génération d'images à partir d'un modèle de diffusion entraîné."""
import argparse
import torch
from diffusion_model import UNet
from diffusion_model.noise import p_sample_loop


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = UNet(img_channels=3, base_channel=64, timesteps=args.timesteps).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    with torch.no_grad():
        out = p_sample_loop(model, (args.num_samples, 3, args.image_size, args.image_size), args.timesteps, device)
    out = (out.clamp(-1, 1) + 1) / 2.0

    torch.save(out, args.output)
    print(f"Échantillons générés: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génère des images par diffusion")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, default="samples.pt")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    main(args)
