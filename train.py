"""Script d'entraînement du modèle de diffusion."""
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
    # Initialisation du modèle principal (architecture UNet moderne).
    model = UNet(img_channels=3, base_channel=64, timesteps=args.timesteps)
    # Chargement des données d'entraînement (CIFAR-10 ou sous-ensemble).
    dataloader = get_dataloader(
        batch_size=args.batch_size,
        image_size=args.image_size,
        train=True,
        num_workers=args.num_workers,
        subset_size=args.subset_size,
    )
    # Dataloader de validation optionnel pour suivre une courbe de généralisation.
    val_dataloader = None
    if args.val_subset_size and args.val_subset_size > 0:
        val_dataloader = get_dataloader(
            batch_size=args.val_batch_size,
            image_size=args.image_size,
            train=False,
            num_workers=args.num_workers,
            subset_size=args.val_subset_size,
        )

    # Préparation des dossiers de sortie (checkpoints + samples intermédiaires).
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_output_dir, exist_ok=True)

    if args.resume:
        # Si --resume est fourni, on tente de reprendre depuis ce checkpoint.
        if os.path.isfile(args.resume):
            # Reprise robuste depuis un checkpoint potentiellement Legacy.
            model = load_model_from_checkpoint(args.resume, device, args.timesteps)
            print(f"Reprise depuis le checkpoint: {args.resume}")
        else:
            raise FileNotFoundError(f"Checkpoint de reprise introuvable: {args.resume}")

    effective_train_timesteps = min(args.timesteps, model.timesteps)
    if effective_train_timesteps != args.timesteps:
        print(
            f"Timesteps d'entrainement ajustes de {args.timesteps} a {effective_train_timesteps} "
            "pour correspondre au modele charge."
        )

    effective_sample_timesteps = min(args.sample_timesteps, model.timesteps)
    if effective_sample_timesteps != args.sample_timesteps:
        print(
            f"Timesteps de sampling ajustes de {args.sample_timesteps} a {effective_sample_timesteps} "
            "pour correspondre au modele charge."
        )

    def save_epoch_samples(epoch):
        # Passage en mode évaluation pendant la génération d'aperçus.
        model.eval()
        with torch.no_grad():
            # Génération d'images de monitoring à la fin de certaines epochs.
            samples = p_sample_loop(
                model,
                (args.sample_num, 3, args.image_size, args.image_size),
                effective_sample_timesteps,
                device,
            )
        # Re-normalisation des pixels pour sauvegarde image.
        samples = (samples.clamp(-1, 1) + 1) / 2.0
        image_path = os.path.join(args.sample_output_dir, f"epoch_{epoch}_samples.png")
        # Sauvegarde PNG + tensor .pt pour réutilisation ultérieure.
        save_image(samples, image_path, nrow=min(8, args.sample_num))
        torch.save(samples, os.path.join(args.sample_output_dir, f"epoch_{epoch}_samples.pt"))
        print(f"Échantillons d'entrainement sauvegardés: {image_path}")
        # Retour en mode entraînement pour l'epoch suivante.
        model.train()

    # Boucle d'entraînement orchestrée par la classe Trainer.
    # Initialisation de l'orchestrateur d'entraînement (optimiseur + loss interne).
    trainer = Trainer(model, dataloader, lr=args.lr, device=device)
    history = trainer.train(
        epochs=args.epochs,
        timesteps=effective_train_timesteps,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
        sample_interval=args.sample_interval,
        sample_fn=save_epoch_samples,
        val_dataloader=val_dataloader,
        val_num_batches=args.val_num_batches,
    )

    if args.output:
        # Sauvegarde finale des poids du modèle entraîné.
        torch.save(model.state_dict(), args.output)
        print(f"Modèle final sauvegardé dans {args.output}")

    if args.history_output:
        os.makedirs(os.path.dirname(args.history_output) or ".", exist_ok=True)
        with open(args.history_output, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        print(f"Historique d'entrainement sauvegarde: {args.history_output}")

    return history


if __name__ == "__main__":
    # Définition des arguments CLI du script d'entraînement.
    parser = argparse.ArgumentParser(description="Entraîne un modèle de diffusion simple")
    # Nombre total d'epochs.
    parser.add_argument("--epochs", type=int, default=10)
    # Taille des batches d'entraînement.
    parser.add_argument("--batch-size", type=int, default=64)
    # Learning rate de l'optimiseur Adam.
    parser.add_argument("--lr", type=float, default=2e-4)
    # Nombre d'étapes de diffusion pour l'entraînement.
    parser.add_argument("--timesteps", type=int, default=1000)
    # Résolution des images d'entrée/sortie.
    parser.add_argument("--image-size", type=int, default=32)
    # Taille max du sous-ensemble de train (optionnel).
    parser.add_argument("--subset-size", type=int, default=None, help="Taille du sous-ensemble d'entraînement pour des runs rapides")
    # Nombre de workers DataLoader.
    parser.add_argument("--num-workers", type=int, default=2)
    # Taille du batch de validation pour la courbe val_loss.
    parser.add_argument("--val-batch-size", type=int, default=64)
    # Taille du sous-ensemble de validation (0 pour desactiver val_loss).
    parser.add_argument("--val-subset-size", type=int, default=1000)
    # Nombre max de batches de validation par epoch.
    parser.add_argument("--val-num-batches", type=int, default=10)
    # Fréquence de sauvegarde des checkpoints (en epochs).
    parser.add_argument("--checkpoint-interval", type=int, default=0, help="Enregistre un checkpoint tous les n epochs")
    # Dossier de sortie des checkpoints.
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints")
    # Fréquence de génération d'échantillons de monitoring.
    parser.add_argument("--sample-interval", type=int, default=0, help="Génère un échantillon tous les n epochs")
    # Nombre d'images générées lors du monitoring.
    parser.add_argument("--sample-num", type=int, default=4)
    # Timesteps de sampling pour le monitoring intermédiaire.
    parser.add_argument("--sample-timesteps", type=int, default=None, help="Nombre de timesteps à utiliser pour les échantillons de validation")
    # Dossier de sortie des images/tenseurs de monitoring.
    parser.add_argument("--sample-output-dir", type=str, default="outputs/train_samples")
    # Chemin de sauvegarde du modèle final.
    parser.add_argument("--output", type=str, default="diffusion_model.pth")
    # Fichier JSON de sortie de l'historique train/val.
    parser.add_argument("--history-output", type=str, default="")
    # Checkpoint à reprendre au démarrage.
    parser.add_argument("--resume", type=str, default=None, help="Chemin du checkpoint à reprendre")
    # Forcer l'exécution CPU.
    parser.add_argument("--cpu", action="store_true")
    # Parsing effectif des arguments CLI.
    args = parser.parse_args()

    # Par défaut, l'échantillonnage de validation utilise les mêmes timesteps que l'entraînement.
    if args.sample_timesteps is None:
        args.sample_timesteps = args.timesteps

    main(args)
