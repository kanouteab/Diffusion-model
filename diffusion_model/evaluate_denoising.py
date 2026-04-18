def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    model = load_model_from_checkpoint(args.model, device, args.timesteps)
    model.eval()

    dataloader = get_dataloader(
        batch_size=args.batch_size,
        image_size=args.image_size,
        train=False,
        num_workers=args.num_workers,
        subset_size=args.subset_size,
    )

    timestep = min(args.noise_timestep, model.timesteps - 1)

    total_mse = 0.0
    total_samples = 0

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(device)

            t = torch.full((x.size(0),), timestep, device=device, dtype=torch.long)

            x_t, _ = q_sample(
                x,
                t,
                model.sqrt_alphas_cumprod,
                model.sqrt_one_minus_alphas_cumprod,
            )

            x_restored = denoise_one_step(model, x_t, t)

            mse = F.mse_loss(x_restored, x)
            total_mse += mse.item() * x.size(0)
            total_samples += x.size(0)

            if args.num_batches and idx + 1 >= args.num_batches:
                break

    print(f"MSE moyen: {total_mse / total_samples}")
