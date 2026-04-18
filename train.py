import argparse
import json
import os

import torch
from torchvision.utils import save_image

from diffusion_model import Trainer, UNet, load_model_from_checkpoint
from diffusion_model.noise import p_sample_loop
from diffusion_model.utils import get_dataloader


def main(args):
    # Détection automatique du device, sauf si --cpu force le CPU.
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device utilisé : {device}")

    # Initialisation du modèle principal (architecture UNet actuelle).
    model = UNet(img_channels=3, base_channel=64, timesteps=args.timesteps)

    # Chargement des données d'entraînement.
    dataloader = get_dataloader(
        batch_size=args.batch_size,
        image_size=args.image_size,
        train=True,
        num_workers=args.num_workers,
        subset_size=args.subset_size,
    )

    # Dataloader de validation optionnel.
    val_dataloader = None
    if getattr(args, "val_subset_size", 0):
        val_dataloader = get_dataloader(
            batch_size=getattr(args, "val_batch_size", args.batch_size),
            image_size=args.image_size,
            train=False,
            num_workers=args.num_workers,
            subset_size=args.val_subset_size,
        )

    # Préparation des dossiers de sortie.
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_output_dir, exist_ok=True)

    # Reprise éventuelle depuis un checkpoint.
    if args.resume:
        if os.path.isfile(args.resume):
            model = load_model_from_checkpoint(args.resume, device, args.timesteps)
            print(f"Reprise depuis le checkpoint : {args.resume}")
        else:
            raise FileNotFoundError(f"Checkpoint de reprise introuvable : {args.resume}")

    # Ajustements de timesteps si le checkpoint chargé en impose moins.
    effective_train_timesteps = min(args.timesteps, model.timesteps)
    if effective_train_timesteps != args.timesteps:
        print(
            f"Timesteps d'entraînement ajustés de {args.timesteps} à "
            f"{effective_train_timesteps} pour correspondre au modèle chargé."
        )

    effective_sample_timesteps = min(args.sample_timesteps, model.timesteps)
    if effective_sample_timesteps != args.sample_timesteps:
        print(
            f"Timesteps de sampling ajustés de {args.sample_timesteps} à "
            f"{effective_sample_timesteps} pour correspondre au modèle chargé."
        )

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
        save_image(samples, image_path, nrow=min(8, args.sample_num))
        torch.save(samples, os.path.join(args.sample_output_dir, f"epoch_{epoch}_samples.pt"))
        print(f"Échantillons d'entraînement sauvegardés : {image_path}")
        model.train()

    # Orchestrateur d'entraînement.
    trainer = Trainer(
        model,
        dataloader,
        lr=args.lr,
        device=device,
        weight_decay=getattr(args, "weight_decay", 0.0),
    )

    history = trainer.train(
        epochs=args.epochs,
        timesteps=effective_train_timesteps,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
        sample_interval=args.sample_interval,
        sample_fn=save_epoch_samples,
        val_dataloader=val_dataloader,
        val_num_batches=args.val_num_batches,
        best_model_path=getattr(args, "best_model_output", None),
        grad_clip=getattr(args, "grad_clip", None),
    )

    # Sauvegarde finale du modèle.
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        torch.save(model.state_dict(), args.output)
        print(f"Modèle final sauvegardé dans {args.output}")

    # Sauvegarde de l'historique.
    if args.history_output:
        os.makedirs(os.path.dirname(args.history_output) or ".", exist_ok=True)
        with open(args.history_output, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        print(f"Historique d'entraînement sauvegardé : {args.history_output}")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîne un modèle de diffusion simple")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--val-subset-size", type=int, default=1000)
    parser.add_argument("--val-num-batches", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--sample-interval", type=int, default=0)
    parser.add_argument("--sample-num", type=int, default=4)
    parser.add_argument("--sample-timesteps", type=int, default=None)
    parser.add_argument("--sample-output-dir", type=str, default="outputs/train_samples")
    parser.add_argument("--output", type=str, default="diffusion_model.pth")
    parser.add_argument("--history-output", type=str, default="")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--best-model-output", type=str, default="outputs/best_model.pth")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    if args.sample_timesteps is None:
        args.sample_timesteps = args.timesteps

    main(args)
