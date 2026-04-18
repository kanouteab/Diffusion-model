"""Évaluation simple d'un modèle de diffusion sur CIFAR-10."""
import argparse

import torch
import torch.nn as nn
from diffusion_model import load_model_from_checkpoint
from diffusion_model.noise import q_sample, sample_timesteps
from diffusion_model.utils import get_dataloader


def main(args):
    # Choix du device d'exécution.
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    # Chargement du checkpoint avec fallback UNet/LegacyUNet.
    model = load_model_from_checkpoint(args.model, device, args.timesteps)
    # Sécurité: mode évaluation (désactive dropout/bn training behavior).
    model.eval()
    effective_timesteps = min(args.timesteps, model.timesteps)
    if effective_timesteps != args.timesteps:
        print(
            f"Timesteps ajustes de {args.timesteps} a {effective_timesteps} "
            "pour correspondre au checkpoint."
        )

    # Dataloader de validation CIFAR-10 (ou sous-ensemble).
    dataloader = get_dataloader(
        batch_size=args.batch_size,
        image_size=args.image_size,
        train=False,
        num_workers=args.num_workers,
        subset_size=args.subset_size,
    )

    # MSE standard de prédiction du bruit.
    criterion = nn.MSELoss(reduction="mean")
    total_loss = 0.0
    total_samples = 0

    # Aucune rétropropagation pendant l'évaluation.
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            # Compatibilité avec datasets retournant (x, y) ou x seul.
            if isinstance(batch, (tuple, list)):
                x, _ = batch
            else:
                x = batch
            # Batch déplacé sur device.
            x = x.to(device)
            # Sélection aléatoire des étapes de diffusion.
            t = sample_timesteps(x.size(0), effective_timesteps, device=device)
            # Bruitage de x selon t, puis récupération du bruit cible.
            x_t, noise = q_sample(
                x,
                t,
                model.sqrt_alphas_cumprod,
                model.sqrt_one_minus_alphas_cumprod,
            )
            # Prédiction du bruit et calcul de la perte.
            predicted_noise = model(x_t, t)
            loss = criterion(predicted_noise, noise)
            # Agrégation pondérée par taille du batch.
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
            # Arrêt anticipé optionnel pour évaluation rapide.
            if args.num_batches and idx + 1 >= args.num_batches:
                break

    # Moyenne finale sécurisée même si total_samples == 0.
    avg_loss = total_loss / max(1, total_samples)
    print(f"Évaluation terminée. MSE moyen sur {total_samples} images: {avg_loss:.6f}")
    return {"avg_mse": avg_loss, "num_samples": total_samples}


if __name__ == "__main__":
    # Définition des arguments CLI du script d'évaluation.
    parser = argparse.ArgumentParser(description="Évalue un modèle de diffusion sur CIFAR-10")
    # Checkpoint des poids à évaluer.
    parser.add_argument("--model", type=str, required=True)
    # Nombre de timesteps pour le bruitage q_sample.
    parser.add_argument("--timesteps", type=int, default=1000)
    # Résolution image de l'entrée CIFAR après resize.
    parser.add_argument("--image-size", type=int, default=32)
    # Taille des batches de validation.
    parser.add_argument("--batch-size", type=int, default=64)
    # Taille max du sous-ensemble de validation.
    parser.add_argument("--subset-size", type=int, default=1000, help="Taille du sous-ensemble de validation")
    # Nombre de workers DataLoader.
    parser.add_argument("--num-workers", type=int, default=2)
    # Nombre de batches à parcourir (arrêt anticipé possible).
    parser.add_argument("--num-batches", type=int, default=10, help="Nombre de batches de validation à utiliser")
    # Forcer l'exécution CPU.
    parser.add_argument("--cpu", action="store_true")
    # Parsing effectif des arguments CLI.
    args = parser.parse_args()
    # Point d'entrée principal.
    main(args)
