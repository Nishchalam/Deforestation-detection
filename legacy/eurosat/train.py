import argparse
import torch
from src.dataset import create_dataloaders
from src.models import create_model
from src.training import Trainer


def resolve_device(name: str) -> torch.device:
    name = (name or "auto").lower()
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description="Train a Deforestation Detection model.")
    parser.add_argument("--model", type=str, default="resnet18",
                        help="Model name (lenet, alexnet, vgg16, googlenet, resnet18, resnet50, efficientnetb0)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (default 16 — memory-friendly; original notebooks used 32)")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers")
    parser.add_argument("--scheduler", type=str, default="plateau",
                        choices=["plateau", "step", "cosine", "none"], help="Learning rate scheduler type")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--min_delta", type=float, default=0.0, help="Early stopping min delta")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device to train on. 'auto' uses CUDA if available.")
    parser.add_argument("--amp", action="store_true",
                        help="Enable CUDA automatic mixed precision. Roughly halves VRAM usage and speeds up training on modern GPUs. No effect on CPU.")
    parser.add_argument("--no_tb_histograms", action="store_true",
                        help="Skip logging per-parameter and per-gradient histograms to TensorBoard. Saves significant CPU RAM and disk on large models.")
    parser.add_argument("--pin_memory", action="store_true",
                        help="Force pin_memory=True on dataloaders. Default: only when training on CUDA.")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    pin_memory = True if args.pin_memory else (device.type == "cuda")

    train_loader, val_loader, _ = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    print(f"Creating model: {args.model}")
    model = create_model(args.model, num_classes=10)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler_type=args.scheduler,
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        early_stopping_min_delta=args.min_delta,
        device=device,
        model_name=args.model,
        training_arguments=vars(args),
        use_amp=args.amp,
        log_param_histograms=not args.no_tb_histograms,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.fit()


if __name__ == "__main__":
    main()
