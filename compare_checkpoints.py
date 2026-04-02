"""Compare plusieurs checkpoints de diffusion en générant une grille d'images côte à côte."""
import argparse
import json
import os

import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image

from diffusion_model import load_model_from_checkpoint
from diffusion_model.noise import p_sample_loop, q_sample, sample_timesteps
from diffusion_model.utils import get_dataloader


def concat_grids(grids):
    # On s'assure d'avoir au moins une grille avant concaténation.
    if not grids:
        raise ValueError("Aucune grille à concaténer")
    # dim=2 correspond à la largeur (concaténation côte à côte).
    concat = torch.cat(grids, dim=2)
    return concat


def evaluate_model_mse(model, args, device):
    effective_timesteps = min(args.timesteps, model.timesteps)
    # Dataloader de validation utilisé uniquement pour le scoring quantitatif.
    dataloader = get_dataloader(
        batch_size=args.score_batch_size,
        image_size=args.image_size,
        train=False,
        num_workers=args.score_num_workers,
        subset_size=args.score_subset_size,
    )
    # MSE entre bruit réel et bruit prédit (objectif DDPM).
    criterion = nn.MSELoss(reduction="mean")
    total_loss = 0.0
    total_samples = 0

    # Pas de gradients pendant l'évaluation.
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            # Compatibilité DataLoader: lot (x, y) ou x seul.
            if isinstance(batch, (tuple, list)):
                x, _ = batch
            else:
                x = batch

            # Envoi du batch sur le bon device.
            x = x.to(device)
            # Tirage aléatoire des étapes de diffusion t.
            t = sample_timesteps(x.size(0), effective_timesteps, device=device)
            # Création de x_t bruité + bruit cible à prédire.
            x_t, noise = q_sample(
                x,
                t,
                model.sqrt_alphas_cumprod,
                model.sqrt_one_minus_alphas_cumprod,
            )
            # Prédiction du bruit par le modèle.
            predicted_noise = model(x_t, t)
            loss = criterion(predicted_noise, noise)
            # Accumulation pondérée par la taille du batch.
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

            # Limitation optionnelle du nombre de batches pour accélérer le scoring.
            if args.score_num_batches and idx + 1 >= args.score_num_batches:
                break

    # Moyenne finale robuste même si aucun batch n'a été traité.
    avg_loss = total_loss / max(1, total_samples)
    return avg_loss, total_samples


def main(args):
    # Sélection automatique du device, avec forçage CPU via --cpu.
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    # Une comparaison a du sens avec au moins 2 checkpoints.
    if len(args.checkpoints) < 2:
        raise ValueError("Au moins deux checkpoints sont requis pour une comparaison.")
    # Validation explicite des chemins pour échouer tôt avec un message clair.
    missing_checkpoints = [ckpt for ckpt in args.checkpoints if not os.path.isfile(ckpt)]
    if missing_checkpoints:
        raise FileNotFoundError(
            "Checkpoint(s) introuvable(s): " + ", ".join(missing_checkpoints)
        )

    # Création du dossier de sortie si nécessaire.
    os.makedirs(os.path.dirname(args.output_image) or ".", exist_ok=True)
    generated_grids = []
    score_results = []

    print(f"Device utilisé: {device}")
    print(f"Seed utilisée: {args.seed}")

    for checkpoint in args.checkpoints:
        # Journalisation du checkpoint actuellement en cours de traitement.
        print(f"Génération pour checkpoint: {checkpoint}")
        # Chargement robuste (UNet puis fallback LegacyUNet).
        model = load_model_from_checkpoint(checkpoint, device, args.timesteps)
        effective_timesteps = min(args.timesteps, model.timesteps)
        if effective_timesteps != args.timesteps:
            print(
                f"Timesteps ajustes de {args.timesteps} a {effective_timesteps} "
                f"pour {checkpoint}."
            )
        # Seed réinitialisée à chaque checkpoint pour comparer des tirages équivalents.
        torch.manual_seed(args.seed)
        with torch.no_grad():
            # Génération d'un mini-lot d'images synthétiques.
            samples = p_sample_loop(
                model,
                (args.num_samples, 3, args.image_size, args.image_size),
                effective_timesteps,
                device,
            )
        # Conversion de [-1, 1] vers [0, 1] pour la sauvegarde image.
        samples = (samples.clamp(-1, 1) + 1) / 2.0
        # Grille d'images pour ce checkpoint.
        generated_grids.append(make_grid(samples, nrow=args.num_samples))

        if args.score:
            # Évaluation quantitative optionnelle du checkpoint.
            mse, num_eval_samples = evaluate_model_mse(model, args, device)
            score_results.append(
                {
                    "checkpoint": checkpoint,
                    "mse": mse,
                    "num_eval_samples": num_eval_samples,
                }
            )
            print(f"Score MSE pour {checkpoint}: {mse:.6f} ({num_eval_samples} images)")

    # Assemblage final de toutes les grilles en une seule image horizontale.
    comparison_grid = concat_grids(generated_grids)
    save_image(comparison_grid, args.output_image)
    print(f"Comparaison sauvegardée: {args.output_image}")

    if score_results:
        # Classement: plus la MSE est basse, meilleur est le checkpoint.
        sorted_results = sorted(score_results, key=lambda x: x["mse"])
        print("\nClassement checkpoints (MSE croissant):")
        for rank, result in enumerate(sorted_results, start=1):
            print(
                f"{rank}. {result['checkpoint']} -> {result['mse']:.6f} "
                f"({result['num_eval_samples']} images)"
            )

        if args.score_output:
            # Export machine-readable pour exploitation ultérieure.
            os.makedirs(os.path.dirname(args.score_output) or ".", exist_ok=True)
            with open(args.score_output, "w", encoding="utf-8") as f:
                json.dump(sorted_results, f, indent=2)
            print(f"Scores sauvegardés: {args.score_output}")


if __name__ == "__main__":
    # Définition des arguments CLI du script de comparaison.
    parser = argparse.ArgumentParser(description="Compare plusieurs checkpoints par génération d'images")
    # Liste des checkpoints à comparer.
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True)
    # Image de comparaison finale (grilles concaténées).
    parser.add_argument("--output-image", type=str, default="checkpoint_comparison.png")
    # Nombre de timesteps utilisé pour le sampling.
    parser.add_argument("--timesteps", type=int, default=1000)
    # Taille des images de sortie.
    parser.add_argument("--image-size", type=int, default=32)
    # Nombre d'images synthétiques par checkpoint.
    parser.add_argument("--num-samples", type=int, default=4)
    # Seed pour garantir une comparaison reproductible.
    parser.add_argument("--seed", type=int, default=42)
    # Active le scoring quantitatif MSE en plus de l'image.
    parser.add_argument("--score", action="store_true", help="Calcule aussi un score MSE par checkpoint")
    # Taille de batch pour la phase de scoring.
    parser.add_argument("--score-batch-size", type=int, default=64)
    # Taille max du sous-ensemble de validation utilisé pour scorer.
    parser.add_argument("--score-subset-size", type=int, default=1000)
    # Nombre de workers DataLoader pour le scoring.
    parser.add_argument("--score-num-workers", type=int, default=2)
    # Nombre maximal de batches de validation à scorer.
    parser.add_argument("--score-num-batches", type=int, default=10)
    # Fichier JSON optionnel de sortie des scores triés.
    parser.add_argument("--score-output", type=str, default="", help="Fichier JSON de sortie des scores")
    # Forcer l'exécution CPU.
    parser.add_argument("--cpu", action="store_true")
    # Parsing effectif des arguments CLI.
    args = parser.parse_args()
    # Point d'entrée principal.
    main(args)
