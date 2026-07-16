import time
import os
import torch
from tqdm.auto import tqdm
from typing import Optional, Any

class Trainer:
    """
    A simple, readable PyTorch trainer for EuroSAT classification.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[torch.nn.Module] = None,
        scheduler: Optional[Any] = None,
        epochs: int = 20,
        device: Optional[torch.device] = None,
        save_path: Optional[str] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion or torch.nn.CrossEntropyLoss()
        self.scheduler = scheduler
        self.epochs = epochs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = save_path
        
        self.model.to(self.device)
        self.best_val_acc = 0.0

    def train_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0
        
        limit_batches = os.environ.get("LIMIT_BATCHES")
        if limit_batches:
            limit_batches = int(limit_batches)
            
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {epoch+1}/{self.epochs} [Train]", 
            leave=False, 
            dynamic_ncols=True
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            if limit_batches and batch_idx >= limit_batches:
                break
                
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
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
                acc=f"{100 * running_correct / total:.2f}%"
            )
            
        epoch_loss = running_loss / total
        epoch_acc = running_correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self) -> tuple[float, float]:
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
            
        val_loss = running_loss / total
        val_acc = running_correct / total
        return val_loss, val_acc

    def fit(self) -> dict[str, list[float]]:
        history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": []
        }
        
        print(f"Training started on device: {self.device}")
        start_time = time.time()
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()
            
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                    
            epoch_time = time.time() - epoch_start
            
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            
            print(
                f"Epoch {epoch+1:02d}/{self.epochs:02d} | "
                f"Time: {epoch_time:.1f}s | "
                f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}"
            )
            
            # Save the best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                if self.save_path:
                    checkpoint = {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_acc": val_acc,
                    }
                    torch.save(checkpoint, self.save_path)
                    print(f"Saved best model checkpoint with Val Acc: {val_acc:.4f}")
                    
        total_time = time.time() - start_time
        print(f"Training completed in {total_time/60:.2f} minutes. Best Val Acc: {self.best_val_acc:.4f}")
        return history
