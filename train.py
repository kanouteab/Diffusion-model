"""Script d'entraînement du modèle de diffusion."""
import argparse
import os

import torch
from torchvision.utils import save_image

from diffusion_model import UNet, Trainer
from diffusion_model.noise import p_sample_loop
from diffusion_model.utils import get_dataloaders


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Expérience 4 : modèle plus large
    model = UNet(img_channels=3, base_channel=128, timesteps=args.timesteps)

    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        subset_size=args.subset_size,
        val_split=args.val_split,
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_output_dir, exist_ok=True)

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)

            # compatible avec ancien format (state_dict seul) et nouveau format (dict)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

            print(f"Reprise depuis le checkpoint: {args.resume}")
        else:
            raise FileNotFoundError(f"Checkpoint de reprise introuvable: {args.resume}")

    def save_epoch_samples(epoch, sample_model):
        sample_model.eval()
        with torch.no_grad():
            samples = p_sample_loop(
                sample_model,
                (args.sample_num, 3, args.image_size, args.image_size),
                args.sample_timesteps,
                device,
            )

        samples = (samples.clamp(-1, 1) + 1) / 2.0
        image_path = os.path.join(args.sample_output_dir, f"epoch_{epoch}_samples.png")
        tensor_path = os.path.join(args.sample_output_dir, f"epoch_{epoch}_samples.pt")

        save_image(samples, image_path, nrow=min(8, args.sample_num))
        torch.save(samples, tensor_path)

        print(f"Échantillons d'entraînement sauvegardés: {image_path}")

    trainer = Trainer(
        model,
        train_loader,
        val_dataloader=val_loader,
        lr=args.lr,
        device=device,
        use_ema=True,
        ema_beta=args.ema_beta,
    )

    trainer.train(
        epochs=args.epochs,
        timesteps=args.timesteps,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
        sample_interval=args.sample_interval,
        sample_fn=save_epoch_samples,
    )

    if args.output:
        save_model = trainer.ema_model if getattr(trainer, "ema_model", None) is not None else trainer.model

        torch.save(
            {
                "model_state_dict": save_model.state_dict(),
                "base_channel": 128,
                "timesteps": args.timesteps,
                "use_ema": True,
                "ema_beta": args.ema_beta,
            },
            args.output,
        )
        print(f"Modèle final sauvegardé dans {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîne un modèle de diffusion simple")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=32)

    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Taille du sous-ensemble d'entraînement pour des runs rapides",
    )
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Proportion du dataset utilisée pour la validation",
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Enregistre un checkpoint tous les n epochs",
    )
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints")

    parser.add_argument(
        "--sample-interval",
        type=int,
        default=10,
        help="Génère un échantillon tous les n epochs",
    )
    parser.add_argument("--sample-num", type=int, default=8)
    parser.add_argument(
        "--sample-timesteps",
        type=int,
        default=None,
        help="Nombre de timesteps à utiliser pour les échantillons",
    )
    parser.add_argument("--sample-output-dir", type=str, default="outputs/train_samples")

    parser.add_argument("--ema-beta", type=float, default=0.999)

    parser.add_argument("--output", type=str, default="diffusion_model.pth")
    parser.add_argument("--resume", type=str, default=None, help="Chemin du checkpoint à reprendre")
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()

    if args.sample_timesteps is None:
        args.sample_timesteps = args.timesteps

    main(args)