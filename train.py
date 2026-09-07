import argparse
import torch
from src.dataset import create_dataloaders
from src.models import create_model
from src.training import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train a Deforestation Detection model.")
    parser.add_argument("--model", type=str, default="resnet18",
                        help="Model name (lenet, alexnet, vgg16, googlenet, resnet18, resnet50, efficientnetb0)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Reduce to 8 or 4 if you OOM. VGG16 is the memory-hungry one.")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Input resolution. Drop to 128 or 96 to fit small VRAM.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true",
                        help="Enable pinned memory (only helps CUDA transfer; leave off on CPU-only).")
    parser.add_argument("--amp", action="store_true",
                        help="Mixed-precision training (fp16). CUDA only; silently ignored on CPU.")
    parser.add_argument("--log_histograms", action="store_true",
                        help="Log per-parameter/gradient histograms to TensorBoard. "
                             "Very memory-heavy for VGG16 -- off by default.")
    parser.add_argument("--scheduler", type=str, default="plateau",
                        choices=["plateau", "step", "cosine", "none"])
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--min_delta", type=float, default=0.0, help="Early stopping min delta")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    train_loader, val_loader, _ = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        img_size=args.img_size,
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
        model_name=args.model,
        training_arguments=vars(args),
        amp=args.amp,
        log_param_histograms=args.log_histograms,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.fit()


if __name__ == "__main__":
    main()
