"""Point d'entree unique pour piloter train/evaluate/sample/compare."""
import argparse
import glob
import importlib
import json
import os
import re
from argparse import Namespace

import compare_checkpoints as compare_script
import evaluate as evaluate_script
import sample as sample_script
import train as train_script


def _checkpoint_sort_key(path: str) -> int:
    match = re.search(r"checkpoint_epoch_(\d+)\.pth$", path)
    if match:
        return int(match.group(1))
    return -1


def run_train(args: Namespace) -> None:
    if args.sample_timesteps is None:
        args.sample_timesteps = args.timesteps
    return train_script.main(args)


def run_evaluate(args: Namespace) -> None:
    return evaluate_script.main(args)


def run_sample(args: Namespace) -> None:
    sample_script.main(args)


def run_compare(args: Namespace) -> None:
    compare_script.main(args)


def _save_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _plot_learning_curves(history: dict, output_path: str) -> bool:
    if not history or not history.get("train_loss"):
        print("Courbe non generee: historique train_loss vide.")
        return False
    try:
        module_name = "matplotlib" + ".pyplot"
        plt = importlib.import_module(module_name)
    except ImportError:
        print("Matplotlib indisponible: courbe d'apprentissage non generee.")
        return False

    epochs = list(range(1, len(history["train_loss"]) + 1))
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="train_loss")

    val_loss = history.get("val_loss", [])
    if val_loss:
        plt.plot(epochs[: len(val_loss)], val_loss, marker="s", label="val_loss")

    plt.title("Courbes d'apprentissage")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.grid(alpha=0.3)
    plt.legend()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Courbe d'apprentissage sauvegardee: {output_path}")
    return True


def _plot_performance_report(report: dict, output_path: str) -> bool:
    values = []
    labels = []

    trained = report.get("trained_model")
    if trained:
        labels.append("trained")
        values.append(trained.get("avg_mse", 0.0))

    pretrained = report.get("pretrained_model")
    if pretrained and pretrained.get("avg_mse") is not None:
        labels.append("pretrained")
        values.append(pretrained.get("avg_mse", 0.0))

    if not values:
        print("Graphique performance non genere: metriques manquantes.")
        return False

    try:
        module_name = "matplotlib" + ".pyplot"
        plt = importlib.import_module(module_name)
    except ImportError:
        print("Matplotlib indisponible: graphique performance non genere.")
        return False

    plt.figure(figsize=(7, 5))
    plt.bar(labels, values)
    plt.title("Comparaison de performance (MSE)")
    plt.ylabel("MSE (plus bas = meilleur)")
    plt.grid(axis="y", alpha=0.3)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Graphique de performance sauvegarde: {output_path}")
    return True


def run_all_pipeline(args: Namespace) -> None:
    print("=== Etape 1/4: entrainement ===")
    train_args = Namespace(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        timesteps=args.timesteps,
        image_size=args.image_size,
        subset_size=args.subset_size,
        num_workers=args.num_workers,
        val_batch_size=args.val_batch_size,
        val_subset_size=args.val_subset_size,
        val_num_batches=args.val_num_batches,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
        sample_interval=args.sample_interval,
        sample_num=args.sample_num,
        sample_timesteps=args.sample_timesteps,
        sample_output_dir=args.sample_output_dir,
        output=args.train_output,
        history_output=args.history_output,
        resume=args.resume,
        cpu=args.cpu,
    )
    history = run_train(train_args)
    _plot_learning_curves(history, args.curves_output)

    print("=== Etape 2/4: evaluation ===")
    eval_args = Namespace(
        model=args.train_output,
        timesteps=args.timesteps,
        image_size=args.image_size,
        batch_size=args.eval_batch_size,
        subset_size=args.eval_subset_size,
        num_workers=args.eval_num_workers,
        num_batches=args.eval_num_batches,
        cpu=args.cpu,
    )
    trained_metrics = run_evaluate(eval_args)

    pretrained_metrics = None
    if args.pretrained_model:
        if os.path.isfile(args.pretrained_model):
            print("=== Evaluation modele pre-entraine ===")
            pretrained_eval_args = Namespace(
                model=args.pretrained_model,
                timesteps=args.timesteps,
                image_size=args.image_size,
                batch_size=args.eval_batch_size,
                subset_size=args.eval_subset_size,
                num_workers=args.eval_num_workers,
                num_batches=args.eval_num_batches,
                cpu=args.cpu,
            )
            pretrained_metrics = run_evaluate(pretrained_eval_args)
        else:
            print(f"Modele pre-entraine ignore (introuvable): {args.pretrained_model}")

    report = {
        "trained_model": {
            "path": args.train_output,
            "avg_mse": trained_metrics["avg_mse"],
            "num_samples": trained_metrics["num_samples"],
        },
        "pretrained_model": {
            "path": args.pretrained_model,
            "avg_mse": pretrained_metrics["avg_mse"] if pretrained_metrics else None,
            "num_samples": pretrained_metrics["num_samples"] if pretrained_metrics else None,
        },
        "delta_mse_pretrained_minus_trained": (
            (pretrained_metrics["avg_mse"] - trained_metrics["avg_mse"])
            if pretrained_metrics
            else None
        ),
    }
    _save_json(args.performance_output, report)
    print(f"Rapport de performance sauvegarde: {args.performance_output}")
    _plot_performance_report(report, args.performance_plot)

    print("=== Etape 3/4: generation d'echantillons ===")
    sample_args = Namespace(
        model=args.train_output,
        output=args.sample_output,
        output_image=args.sample_output_image,
        output_gif=args.sample_output_gif,
        gif_frame_interval=args.gif_frame_interval,
        gif_fps=args.gif_fps,
        timesteps=args.timesteps,
        image_size=args.image_size,
        num_samples=args.sample_num,
        seed=args.seed,
        cpu=args.cpu,
    )
    run_sample(sample_args)

    print("=== Etape 4/4: comparaison de checkpoints ===")
    checkpoints = list(args.compare_checkpoints) if args.compare_checkpoints else []
    if not checkpoints:
        pattern = os.path.join(args.checkpoint_dir, "checkpoint_epoch_*.pth")
        discovered = sorted(glob.glob(pattern), key=_checkpoint_sort_key)
        if len(discovered) >= 2:
            checkpoints = [discovered[0], discovered[-1]]
            print(
                "Checkpoints auto-selectionnes pour comparaison: "
                f"{checkpoints[0]} et {checkpoints[1]}"
            )

    if args.include_pretrained_in_compare and args.pretrained_model and os.path.isfile(args.pretrained_model):
        if args.pretrained_model not in checkpoints:
            checkpoints.append(args.pretrained_model)
            print(f"Checkpoint pre-entraine ajoute a la comparaison: {args.pretrained_model}")

    if len(checkpoints) < 2:
        print("Comparaison ignoree: au moins 2 checkpoints sont necessaires.")
        return

    compare_args = Namespace(
        checkpoints=checkpoints,
        output_image=args.compare_output_image,
        timesteps=args.timesteps,
        image_size=args.image_size,
        num_samples=args.compare_num_samples,
        seed=args.seed,
        score=args.compare_score,
        score_batch_size=args.score_batch_size,
        score_subset_size=args.score_subset_size,
        score_num_workers=args.score_num_workers,
        score_num_batches=args.score_num_batches,
        score_output=args.score_output,
        cpu=args.cpu,
    )
    run_compare(compare_args)


def add_train_parser(subparsers) -> None:
    parser = subparsers.add_parser("train", help="Lancer uniquement l'entrainement")
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
    parser.add_argument("--cpu", action="store_true")
    parser.set_defaults(func=run_train)


def add_evaluate_parser(subparsers) -> None:
    parser = subparsers.add_parser("evaluate", help="Lancer uniquement l'evaluation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--subset-size", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    parser.set_defaults(func=run_evaluate)


def add_sample_parser(subparsers) -> None:
    parser = subparsers.add_parser("sample", help="Lancer uniquement le sampling")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, default="samples.pt")
    parser.add_argument("--output-image", type=str, default=None)
    parser.add_argument("--output-gif", type=str, default=None)
    parser.add_argument("--gif-frame-interval", type=int, default=10)
    parser.add_argument("--gif-fps", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.set_defaults(func=run_sample)


def add_compare_parser(subparsers) -> None:
    parser = subparsers.add_parser("compare", help="Lancer uniquement la comparaison")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True)
    parser.add_argument("--output-image", type=str, default="checkpoint_comparison.png")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--score-batch-size", type=int, default=64)
    parser.add_argument("--score-subset-size", type=int, default=1000)
    parser.add_argument("--score-num-workers", type=int, default=2)
    parser.add_argument("--score-num-batches", type=int, default=10)
    parser.add_argument("--score-output", type=str, default="")
    parser.add_argument("--cpu", action="store_true")
    parser.set_defaults(func=run_compare)


def add_all_parser(subparsers) -> None:
    parser = subparsers.add_parser("all", help="Pipeline complet: train + evaluate + sample + compare")
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
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--sample-interval", type=int, default=1)
    parser.add_argument("--sample-num", type=int, default=4)
    parser.add_argument("--sample-timesteps", type=int, default=None)
    parser.add_argument("--sample-output-dir", type=str, default="outputs/train_samples")
    parser.add_argument("--train-output", type=str, default="outputs/unet_trained_final.pth")
    parser.add_argument("--history-output", type=str, default="outputs/learning_history.json")
    parser.add_argument("--curves-output", type=str, default="outputs/learning_curves.png")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--pretrained-model", type=str, default="outputs/unet_trained.pth")
    parser.add_argument("--performance-output", type=str, default="outputs/performance_report.json")
    parser.add_argument("--performance-plot", type=str, default="outputs/performance_comparison.png")

    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-subset-size", type=int, default=1000)
    parser.add_argument("--eval-num-workers", type=int, default=2)
    parser.add_argument("--eval-num-batches", type=int, default=10)

    parser.add_argument("--sample-output", type=str, default="outputs/final_samples.pt")
    parser.add_argument("--sample-output-image", type=str, default="outputs/final_samples.png")
    parser.add_argument("--sample-output-gif", type=str, default=None)
    parser.add_argument("--gif-frame-interval", type=int, default=10)
    parser.add_argument("--gif-fps", type=int, default=5)

    parser.add_argument("--compare-checkpoints", type=str, nargs="*", default=[])
    parser.add_argument("--compare-output-image", type=str, default="outputs/checkpoint_comparison.png")
    parser.add_argument("--compare-num-samples", type=int, default=4)
    parser.add_argument("--compare-score", action="store_true")
    parser.add_argument("--include-pretrained-in-compare", action="store_true")
    parser.add_argument("--score-batch-size", type=int, default=64)
    parser.add_argument("--score-subset-size", type=int, default=1000)
    parser.add_argument("--score-num-workers", type=int, default=2)
    parser.add_argument("--score-num-batches", type=int, default=10)
    parser.add_argument("--score-output", type=str, default="outputs/checkpoint_scores.json")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.set_defaults(func=run_all_pipeline)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Point d'entree principal du projet Diffusion-model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_train_parser(subparsers)
    add_evaluate_parser(subparsers)
    add_sample_parser(subparsers)
    add_compare_parser(subparsers)
    add_all_parser(subparsers)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
