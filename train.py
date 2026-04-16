import argparse
import os

import torch
from torchvision.utils import save_image

from diffusion_model import Trainer, UNet, load_model_from_checkpoint
from diffusion_model.noise import p_sample_loop
from diffusion_model.utils import get_dataloaders


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device utilisé : {device}")

    # Chargement / création du modèle
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Checkpoint introuvable : {args.resume}")

        model = load_model_from_checkpoint(
            checkpoint=args.resume,
            device=device,
            timesteps=args.timesteps,
            base_channel=args.base_channel,
        )
        print(f"Reprise depuis le checkpoint : {args.resume}")
    else:
        model = UNet(
            img_channels=3,
            base_channel=args.base_channel,
            timesteps=args.timesteps,
        ).to(device)

    # Dataloaders
    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        subset_size=args.subset_size,
        val_split=args.val_split,
    )

    # Dossiers de sortie
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_output_dir, exist_ok=True)

    effective_sample_timesteps = args.sample_timesteps or args.timesteps

    def save_epoch_samples(epoch):
        model.eval()
        with torch.no_grad():
            samples = p_sample_loop(
                model,
                (args.sample_num, 3, args.image_size, args.image_size),
                effective_sample_timesteps,
                device,
            )

        samples = (samples.clamp(-1, 1) + 1) / 2.0

        image_path = os.path.join(args.sample_output_dir, f"epoch_{epoch}_samples.png")
        tensor_path = os.path.join(args.sample_output_dir, f"epoch_{epoch}_samples.pt")

        save_image(samples, image_path, nrow=min(8, args.sample_num))
        torch.save(samples, tensor_path)

        print(f"Échantillons sauvegardés : {image_path}")
        model.train()

    # Trainer
    trainer = Trainer(model, train_loader, lr=args.lr, device=device)

    trainer.train(
        epochs=args.epochs,
        timesteps=args.timesteps,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
        sample_interval=args.sample_interval,
        sample_fn=save_epoch_samples,
    )

    # Sauvegarde finale
    if args.output:
        torch.save(model.state_dict(), args.output)
        print(f"Modèle final sauvegardé dans {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîne un modèle de diffusion simple")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--base-channel", type=int, default=64)

    parser.add_argument("--subset-size", type=int, default=None, help="Taille du sous-ensemble d'entraînement")
    parser.add_argument("--val-split", type=float, default=0.0, help="Proportion de validation")
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--checkpoint-interval", type=int, default=0, help="Sauvegarde un checkpoint tous les n epochs")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints")

    parser.add_argument("--sample-interval", type=int, default=0, help="Génère des échantillons tous les n epochs")
    parser.add_argument("--sample-num", type=int, default=8)
    parser.add_argument("--sample-timesteps", type=int, default=None)
    parser.add_argument("--sample-output-dir", type=str, default="outputs/train_samples")

    parser.add_argument("--output", type=str, default="diffusion_model.pth")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    main(args)