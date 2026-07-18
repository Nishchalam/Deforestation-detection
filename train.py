import argparse
import torch
from src.dataset import create_dataloaders
from src.models import create_model
from src.training import Trainer

def main():
    parser = argparse.ArgumentParser(description="Train a Deforestation Detection model.")
    parser.add_argument("--model", type=str, default="resnet18", 
                        help="Model name (lenet, alexnet, vgg16, googlenet, resnet18, resnet50, efficientnetb0)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers")
    parser.add_argument("--scheduler", type=str, default="plateau", choices=["plateau", "step", "cosine", "none"], help="Learning rate scheduler type")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--min_delta", type=float, default=0.0, help="Early stopping min delta")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    # Create loaders
    train_loader, val_loader, _ = create_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers)
    
    # Create model
    print(f"Creating model: {args.model}")
    model = create_model(args.model, num_classes=10)
    
    # Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Trainer
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
        training_arguments=vars(args)
    )
    
    # Resume if requested
    if args.resume:
        trainer.load_checkpoint(args.resume)
        
    # Fit
    trainer.fit()

if __name__ == "__main__":
    main()
