import argparse
import torch
from src.dataset import create_dataloaders
from src.models import create_model
from src.evaluation import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Evaluate a Deforestation Detection model.")
    parser.add_argument("--model", type=str, default="resnet18", help="Model name")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model weights checkpoint (.pth)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers")
    args = parser.parse_args()

    # Load test dataloader
    _, _, test_loader = create_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers)
    
    # Create model and load checkpoint
    model = create_model(args.model, num_classes=10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    criterion = torch.nn.CrossEntropyLoss()
    
    print(f"Evaluating {args.model} on test set...")
    metrics, _, _ = evaluate_model(model, test_loader, criterion, device)
    
    print("\n" + "="*40)
    print("               EVALUATION RESULTS")
    print("="*40)
    print(f"Test Accuracy    : {metrics['accuracy']:.4f}")
    print(f"Precision        : {metrics['precision']:.4f}")
    print(f"Recall           : {metrics['recall']:.4f}")
    print(f"F1-Score         : {metrics['f1']:.4f}")
    print(f"Latency per Image: {metrics['inference_time_per_image'] * 1000:.3f} ms")
    print(f"Throughput       : {metrics['images_per_second']:.2f} images/sec")
    print("="*40)

if __name__ == "__main__":
    main()
