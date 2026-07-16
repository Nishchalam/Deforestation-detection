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
    parser.add_argument("--save_path", type=str, default="best_model.pth", help="Path to save best model checkpoint")
    args = parser.parse_args()

    # Create loaders
    train_loader, val_loader, _ = create_dataloaders(batch_size=args.batch_size)
    
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
        epochs=args.epochs,
        save_path=args.save_path
    )
    
    # Fit
    trainer.fit()

if __name__ == "__main__":
    main()
