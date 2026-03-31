"""Script d'entraînement du modèle de diffusion."""
import argparse
import torch
from diffusion_model import UNet, Trainer
from diffusion_model.utils import get_dataloader


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = UNet(img_channels=3, base_channel=64, timesteps=args.timesteps)
    dataloader = get_dataloader(batch_size=args.batch_size, image_size=args.image_size, train=True)
    trainer = Trainer(model, dataloader, lr=args.lr, device=device)
    trainer.train(epochs=args.epochs, timesteps=args.timesteps)

    if args.output:
        torch.save(model.state_dict(), args.output)
        print(f"Modèle sauvegardé dans {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîne un modèle de diffusion simple")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--output", type=str, default="diffusion_model.pth")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    main(args)
