"""
trainer.py

Refactored Generic Trainer using Callbacks and ExperimentLogger.
"""

import time
import torch
from tqdm.auto import tqdm
from typing import List, Dict, Any, Optional

from src.training.losses import get_loss
from src.training.utils import format_time
from src.training.callbacks import Callback

class Trainer:
    """
    Generic Trainer implementing a callback system and logging.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        epochs: int = 20,
        device: Optional[torch.device] = None,
        loss_name: str = "cross_entropy",
        scheduler: Optional[Any] = None,
        callbacks: Optional[List[Callback]] = None,
        logger: Optional[Any] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = get_loss(loss_name)
        self.scheduler = scheduler
        self.callbacks = callbacks or []
        self.logger = logger
        self.stop_training = False

        if self.logger:
            self.logger.log_info(f"Trainer initialized on device: {self.device}")

    def _trigger_callbacks(self, hook: str, *args, **kwargs):
        for callback in self.callbacks:
            method = getattr(callback, hook, None)
            if callable(method):
                method(*args, **kwargs)

    def train_one_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0

        progress_bar = tqdm(self.train_loader, desc=f"Train Epoch {epoch+1}", leave=False, dynamic_ncols=True)

        import os
        limit_batches = os.environ.get("LIMIT_BATCHES")
        if limit_batches:
            limit_batches = int(limit_batches)

        for batch_idx, batch in enumerate(progress_bar):
            if limit_batches and batch_idx >= limit_batches:
                break
            self._trigger_callbacks("on_batch_begin", batch_idx)

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

            progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100 * running_correct / total:.2f}%")
            self._trigger_callbacks("on_batch_end", batch_idx, {"loss": loss.item()})

        epoch_loss = running_loss / total
        epoch_acc = running_correct / total
        return epoch_loss, epoch_acc
    
    def validate(self) -> tuple[float, float]:
        self.model.eval()
        running_loss = 0.0
        running_correct = 0
        total = 0

        progress_bar = tqdm(self.val_loader, desc="Validation", leave=False, dynamic_ncols=True)

        import os
        limit_batches = os.environ.get("LIMIT_BATCHES")
        if limit_batches:
            limit_batches = int(limit_batches)

        with torch.no_grad():
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

                progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100 * running_correct / total:.2f}%")

        epoch_loss = running_loss / total
        epoch_acc = running_correct / total
        return epoch_loss, epoch_acc
    
    def fit(self):
        if self.logger:
            self.logger.log_info("Starting training...")
            
        start_time = time.time()
        self._trigger_callbacks("on_train_begin")

        for epoch in range(self.epochs):
            if self.stop_training:
                break
                
            self._trigger_callbacks("on_epoch_begin", epoch)
            
            epoch_start = time.time()
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate()
            
            # Step scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            logs = {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "model": self.model,
                "optimizer": self.optimizer,
                "scheduler": self.scheduler,
            }
            
            self._trigger_callbacks("on_epoch_end", epoch, logs)
            
            # Check for early stopping in callbacks
            for cb in self.callbacks:
                if getattr(cb, "stop_training", False):
                    self.stop_training = True
                    if self.logger:
                        self.logger.log_info(f"Early stopping triggered at epoch {epoch+1}")

            epoch_time = time.time() - epoch_start
            if self.logger:
                self.logger.log_info(
                    f"Epoch {epoch+1}/{self.epochs} - {format_time(epoch_time)} - "
                    f"train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - "
                    f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
                )

        self._trigger_callbacks("on_train_end")
        
        total_time = time.time() - start_time
        if self.logger:
            self.logger.log_info(f"Training completed in {format_time(total_time)}.")