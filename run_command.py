import subprocess

command = [
    "python", "run_all.py", "all",

    "--epochs", "30",
    "--batch-size", "64",
    "--lr", "1e-4",
    "--timesteps", "1000",
    "--image-size", "32",

    "--subset-size", "50000",
    "--val-subset-size", "5000",
    "--eval-subset-size", "1000",

    "--num-workers", "2",
    "--val-batch-size", "64",
    "--val-num-batches", "20",

    "--checkpoint-interval", "5",
    "--checkpoint-dir", "outputs/checkpoints",
    "--sample-interval", "5",
    "--sample-num", "8",
    "--sample-timesteps", "1000",
    "--sample-output-dir", "outputs/train_samples",

    "--train-output", "outputs/unet_trained_final.pth",
    "--history-output", "outputs/learning_history.json",
    "--curves-output", "outputs/learning_curves.png",
    "--best-model-output", "outputs/best_model.pth",

    "--weight-decay", "1e-4",
    "--grad-clip", "0.5",

    "--performance-output", "outputs/performance_report.json",
    "--performance-plot", "outputs/performance_comparison.png",
    "--eval-batch-size", "64",
    "--eval-num-workers", "2",
    "--eval-num-batches", "20",

    "--noise-timestep", "300",
    "--denoising-metrics-output", "outputs/denoising_metrics.json",
    "--denoising-metrics-image", "outputs/noisy_vs_restored_grid.png",
    "--denoising-plot", "outputs/denoising_comparison.png",
    "--denoising-examples-dir", "outputs/denoising_examples",

    "--sample-output", "outputs/final_samples.pt",
    "--sample-output-image", "outputs/final_samples.png",

    "--compare-output-image", "outputs/checkpoint_comparison.png",
    "--compare-num-samples", "8",
    "--score-output", "outputs/checkpoint_scores.json",

    "--seed", "42"
]

subprocess.run(command)
