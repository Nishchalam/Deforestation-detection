import argparse
from pathlib import Path

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.dataloader import create_dataloaders
from src.models import (
    LeNet, AlexNet, VGG16, GoogLeNet, ResNet18, ResNet50, EfficientNetB0
)
from src.training.trainer import Trainer
from src.training.utils import set_seed, count_parameters

# Registry for available models
MODELS = {
    "lenet": LeNet,
    "alexnet": AlexNet,
    "vgg16": VGG16,
    "googlenet": GoogLeNet,
    "resnet18": ResNet18,
    "resnet50": ResNet50,
    "efficientnet-b0": EfficientNetB0,
    # Future models will be added here
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Land Cover Classification Model"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="lenet", 
        choices=MODELS.keys(), 
        help="Model architecture"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=20, 
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32, 
        help="Batch size"
    )
    
    parser.add_argument(
        "--lr", 
        type=float, 
        default=1e-3, 
        help="Learning rate"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--data-root", 
        type=str, 
        default="data/raw/EuroSAT", 
        help="Root directory containing images"
    )
    
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None, 
        help="Path to checkpoint to resume from"
    )
    
    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()
    
    # 1. Reproducibility
    set_seed(args.seed)
    
    # 2. Dataloaders
    print("Initializing dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
    )
    
    # 3. Model
    print(f"Initializing {args.model}...")
    model_class = MODELS[args.model]
    model = model_class(in_channels=3, num_classes=10)
    print(f"Total trainable parameters: {count_parameters(model)}")
    
    # 4. Trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = f"outputs/logs/{args.model}"
    checkpoint_dir = f"outputs/checkpoints/{args.model}"
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
    )
    
    # Setup learning rate scheduler
    trainer.scheduler = ReduceLROnPlateau(
        trainer.optimizer, 
        mode="min", 
        patience=3, 
        factor=0.1
    )
    
    # 5. Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed at epoch {start_epoch}")
    
    # 6. Training Loop
    try:
        history = trainer.fit()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        trainer.close()
        
    print("\nTraining script completed successfully.")


if __name__ == "__main__":
    main()
