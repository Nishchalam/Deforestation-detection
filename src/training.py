import time
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from typing import Optional, Any, Dict
from pathlib import Path
from src.evaluation import evaluate_model

# Project Root Helper
PROJECT_ROOT = Path(__file__).resolve().parent.parent

class Trainer:
    """
    A robust, reproducible PyTorch trainer for EuroSAT land-cover classification.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[torch.nn.Module] = None,
        scheduler_type: str = "plateau",
        epochs: int = 20,
        early_stopping_patience: int = 3,
        early_stopping_min_delta: float = 0.0,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "outputs/checkpoints",
        tensorboard_dir: str = "outputs/logs",
        model_name: str = "model",
        training_arguments: Optional[Dict[str, Any]] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion or torch.nn.CrossEntropyLoss()
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.training_arguments = training_arguments
        
        # Setup directories
        self.run_dir = PROJECT_ROOT / checkpoint_dir / model_name
        self.log_dir = PROJECT_ROOT / tensorboard_dir / model_name
        
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Scheduler
        self.scheduler_type = scheduler_type
        if scheduler_type == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", patience=2, factor=0.1
            )
        elif scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=5, gamma=0.1
            )
        elif scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs
            )
        else:
            self.scheduler = None

        self.model.to(self.device)
        
        # Initialize metrics tracker
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "learning_rate": [], "epoch": [],
            "training_time": [], "validation_time": []
        }
        
        # Initialize TensorBoard Writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Saves checkpoints to self.run_dir."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "history": self.history,
            "best_metric": {
                "val_loss": self.best_val_loss,
                "val_acc": self.best_val_acc
            },
            "training_arguments": self.training_arguments or {}
        }
        
        # Save last checkpoint
        last_path = self.run_dir / "last_model.pth"
        torch.save(checkpoint, last_path)
        
        if is_best:
            best_path = self.run_dir / "best_model.pth"
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Loads state checkpoint to resume training."""
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        device = self.device
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict") is not None:
            try:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception as e:
                print(f"Warning: Could not load scheduler state dict: {e}")
            
        self.start_epoch = checkpoint["epoch"] + 1
        self.history = checkpoint["history"]
        self.best_val_loss = checkpoint["best_metric"]["val_loss"]
        self.best_val_acc = checkpoint["best_metric"]["val_acc"]
        
        print(f"Resumed from epoch {self.start_epoch}. Best Val Loss: {self.best_val_loss:.4f}, Best Val Acc: {self.best_val_acc:.4f}")

    def train_epoch(self, epoch: int) -> tuple[float, float, float]:
        """Runs one epoch of training."""
        self.model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0
        
        limit_batches = os.environ.get("LIMIT_BATCHES")
        if limit_batches:
            limit_batches = int(limit_batches)
            
        current_lr = self.optimizer.param_groups[0]['lr']
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {epoch+1}/{self.epochs} [Train]", 
            leave=False, 
            dynamic_ncols=True
        )
        
        epoch_start = time.time()
        for batch_idx, batch in enumerate(progress_bar):
            if limit_batches and batch_idx >= limit_batches:
                break
                
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # Log training images grid during first epoch
            if epoch == self.start_epoch and batch_idx == 0:
                try:
                    grid = torchvision.utils.make_grid(images[:8])
                    self.writer.add_image("Training_Images", grid, 0)
                except Exception as e:
                    pass
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += batch_size
            
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}", 
                acc=f"{100 * running_correct / total:.2f}%",
                lr=f"{current_lr:.2e}"
            )
            
        epoch_time = time.time() - epoch_start
        epoch_loss = running_loss / max(1, total)
        epoch_acc = running_correct / max(1, total)
        return epoch_loss, epoch_acc, epoch_time

    @torch.no_grad()
    def validate(self) -> tuple[float, float, float]:
        """Runs evaluation over validation set."""
        self.model.eval()
        running_loss = 0.0
        running_correct = 0
        total = 0
        
        limit_batches = os.environ.get("LIMIT_BATCHES")
        if limit_batches:
            limit_batches = int(limit_batches)
            
        progress_bar = tqdm(
            self.val_loader, 
            desc="[Validation]", 
            leave=False, 
            dynamic_ncols=True
        )
        
        epoch_start = time.time()
        for batch_idx, batch in enumerate(progress_bar):
            if limit_batches and batch_idx >= limit_batches:
                break
                
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += batch_size
            
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}", 
                acc=f"{100 * running_correct / total:.2f}%"
            )
            
        epoch_time = time.time() - epoch_start
        val_loss = running_loss / max(1, total)
        val_acc = running_correct / max(1, total)
        return val_loss, val_acc, epoch_time

    def log_predictions(self, epoch: int):
        """Logs prediction samples to TensorBoard."""
        try:
            self.model.eval()
            val_batch = next(iter(self.val_loader))
            images = val_batch["image"][:6].to(self.device)
            labels = val_batch["label"][:6].cpu().numpy()
            with torch.no_grad():
                outputs = self.model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
            
            fig, axes = plt.subplots(1, 6, figsize=(15, 3))
            classes = [
                "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
                "Industrial", "Pasture", "PermanentCrop", "Residential",
                "River", "SeaLake"
            ]
            for i in range(min(6, len(images))):
                img = val_batch["image"][i].permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
                axes[i].imshow(img)
                color = "green" if preds[i] == labels[i] else "red"
                pred_class = classes[preds[i]] if preds[i] < len(classes) else str(preds[i])
                true_class = classes[labels[i]] if labels[i] < len(classes) else str(labels[i])
                axes[i].set_title(f"Pred: {pred_class}\nTrue: {true_class}", color=color, fontsize=10)
                axes[i].axis("off")
            plt.tight_layout()
            self.writer.add_figure("Example_Predictions", fig, epoch + 1)
            plt.close(fig)
        except Exception:
            pass

    def log_confusion_matrix(self, epoch: int):
        """Logs confusion matrix figure to TensorBoard."""
        try:
            metrics, _, _ = evaluate_model(self.model, self.val_loader, self.criterion, self.device)
            classes = [
                "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
                "Industrial", "Pasture", "PermanentCrop", "Residential",
                "River", "SeaLake"
            ]
            cm = np.array(metrics["confusion_matrix"])
            fig, ax = plt.subplots(figsize=(10, 8))
            import seaborn as sns
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
            ax.set_title("Confusion Matrix")
            ax.set_ylabel("True Class")
            ax.set_xlabel("Predicted Class")
            plt.tight_layout()
            self.writer.add_figure("Confusion_Matrix", fig, epoch + 1)
            plt.close(fig)
        except Exception:
            pass

    def generate_summary(self, total_time: float, early_stopped: bool):
        """Generates training_summary.md file."""
        val_losses = self.history["val_loss"]
        val_accs = self.history["val_acc"]
        
        if len(val_losses) == 0:
            return
            
        best_epoch_idx = int(np.argmin(val_losses))
        best_epoch = best_epoch_idx + 1
        best_loss = val_losses[best_epoch_idx]
        best_acc = val_accs[best_epoch_idx]
        
        summary_content = f"""# Training Summary: {self.model_name}

- **Total Epochs Run**: {len(self.history["epoch"])}
- **Best Epoch**: {best_epoch}
- **Training Time**: {total_time/60:.2f} minutes
- **Best Validation Loss**: {best_loss:.4f}
- **Best Validation Accuracy**: {best_acc*100:.2f}%
- **Early Stopped**: {"Yes" if early_stopped else "No"}
- **Learning Rate Schedule**: {self.scheduler_type} (Initial LR: {self.history["learning_rate"][0]:.2e} -> Final LR: {self.history["learning_rate"][-1]:.2e})
- **Checkpoint Paths**:
  - Best Checkpoint: `{self.run_dir / "best_model.pth"}`
  - Last Checkpoint: `{self.run_dir / "last_model.pth"}`
"""
        summary_path = self.run_dir / "training_summary.md"
        with open(summary_path, "w") as f:
            f.write(summary_content)
        print(f"Generated training summary at {summary_path}")

    def fit(self) -> dict[str, list[float]]:
        """Trains the model for self.epochs."""
        print(f"Training started on device: {self.device}")
        start_time = time.time()
        early_stopped = False
        
        for epoch in range(self.start_epoch, self.epochs):
            train_loss, train_acc, train_time = self.train_epoch(epoch)
            val_loss, val_acc, val_time = self.validate()
            
            # Step scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                    
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["learning_rate"].append(current_lr)
            self.history["epoch"].append(epoch + 1)
            self.history["training_time"].append(train_time)
            self.history["validation_time"].append(val_time)
            
            # Save history to json
            with open(self.run_dir / "history.json", "w") as f:
                json.dump(self.history, f, indent=4)
                
            # Log to TensorBoard
            self.writer.add_scalar("Loss/Train", train_loss, epoch + 1)
            self.writer.add_scalar("Loss/Val", val_loss, epoch + 1)
            self.writer.add_scalar("Accuracy/Train", train_acc, epoch + 1)
            self.writer.add_scalar("Accuracy/Val", val_acc, epoch + 1)
            self.writer.add_scalar("LearningRate", current_lr, epoch + 1)
            
            for name, param in self.model.named_parameters():
                self.writer.add_histogram(f"Parameters/{name}", param, epoch + 1)
                if param.grad is not None:
                    self.writer.add_histogram(f"Gradients/{name}", param.grad, epoch + 1)
            
            self.log_predictions(epoch)
            
            # Early stopping check and status reporting
            print(f"\nEpoch {epoch+1}")
            
            if val_loss < self.best_val_loss - self.early_stopping_min_delta:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                print("Validation Loss Improved")
            else:
                self.patience_counter += 1
                self.save_checkpoint(epoch, is_best=False)
                print(f"No Improvement ({self.patience_counter} / {self.early_stopping_patience})")
                
            print(f"Current Learning Rate: {current_lr:.2e}")
            
            if self.patience_counter >= self.early_stopping_patience:
                print("Early stopping triggered.")
                early_stopped = True
                break
                
        # Finalize TensorBoard logs
        self.log_confusion_matrix(epoch)
        self.writer.close()
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time/60:.2f} minutes. Best Val Loss: {self.best_val_loss:.4f}")
        
        self.generate_summary(total_time, early_stopped)
        return self.history
