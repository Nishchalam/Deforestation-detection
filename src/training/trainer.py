"""
trainer.py

Generic training engine for image classification models.

Features
--------
- CPU / GPU automatic selection
- Training loop
- Validation loop
- TensorBoard logging
- Checkpoint saving
- Resume training
- Learning-rate scheduler
- Progress bars
- History tracking
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.training.metrics import accuracy
from src.training.losses import get_loss


class Trainer:

    """
    Generic Trainer.

    Parameters
    ----------
    model : nn.Module

    train_loader : DataLoader

    val_loader : DataLoader

    epochs : int

    learning_rate : float

    device : torch.device | None

    loss_name : str

    log_dir : str

    checkpoint_dir : str
    """

    def __init__(self,model,train_loader,val_loader,epochs=20,learning_rate=1e-3,device=None,loss_name="cross_entropy",log_dir="outputs/logs",checkpoint_dir="outputs/checkpoints",scheduler=None,):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = (device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.criterion = get_loss(loss_name)
        self.optimizer = optim.Adam(self.model.parameters(),lr=self.learning_rate,)
        self.scheduler = scheduler
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True,exist_ok=True,)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True,exist_ok=True,)
        self.best_model_path = (self.checkpoint_dir/ "best_model.pth")
        self.last_model_path = (self.checkpoint_dir/ "last_model.pth")
        self.history = {

            "train_loss": [],

            "validation_loss": [],

            "train_accuracy": [],

            "validation_accuracy": [],

            "learning_rate": [],

        }

        self.best_validation_accuracy = 0.0
        print("=" * 60)
        print("Trainer initialized")
        print(f"Device : {self.device}")
        print(f"Epochs : {self.epochs}")
        print(f"Learning Rate : {self.learning_rate}")
        print("=" * 60)

    def train_one_epoch(self):

        self.model.train()

        running_loss = 0.0
        running_correct = 0
        total = 0

        progress_bar = tqdm(
            self.train_loader,
            desc="Training",
            leave=False,
            dynamic_ncols=True,
        )

        for batch in progress_bar:

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
    
    def validate(self):

        self.model.eval()

        running_loss = 0.0
        running_correct = 0
        total = 0

        progress_bar = tqdm(
            self.val_loader,
            desc="Validation",
            leave=False,
            dynamic_ncols=True,
        )

        with torch.no_grad():

            for batch in progress_bar:

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

        epoch_loss = running_loss / total
        epoch_acc = running_correct / total

        return epoch_loss, epoch_acc
    
    def fit(self):

        print("\nStarting training...\n")

        for epoch in range(self.epochs):

            print(f"\nEpoch [{epoch+1}/{self.epochs}]")

            train_loss, train_acc = self.train_one_epoch()

            val_loss, val_acc = self.validate()

            if self.scheduler is not None:

                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["validation_loss"].append(val_loss)
            self.history["train_accuracy"].append(train_acc)
            self.history["validation_accuracy"].append(val_acc)
            self.history["learning_rate"].append(current_lr)

            self.writer.add_scalar(
                "Loss/Train",
                train_loss,
                epoch,
            )

            self.writer.add_scalar(
                "Loss/Validation",
                val_loss,
                epoch,
            )

            self.writer.add_scalar(
                "Accuracy/Train",
                train_acc,
                epoch,
            )

            self.writer.add_scalar(
                "Accuracy/Validation",
                val_acc,
                epoch,
            )

            self.writer.add_scalar(
                "Learning Rate",
                current_lr,
                epoch,
            )

            if val_acc > self.best_validation_accuracy:

                self.best_validation_accuracy = val_acc

                self.save_checkpoint(
                    self.best_model_path,
                    epoch,
                )

                best = "✓"
            else:
                best = ""

            self.save_checkpoint(
                self.last_model_path,
                epoch,
            )

            print(
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc*100:.2f}% "
                f"{best}"
            )

        print("\nTraining completed.")

        self.writer.flush()

        return self.history

    def save_checkpoint(self, path, epoch):

        checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": self.model.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict(),
        "history": self.history,
        "best_validation_accuracy": self.best_validation_accuracy,
        "config": {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "device": str(self.device),
        }
    }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)


    def load_checkpoint(self, path):

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        if (
            self.scheduler is not None and
            "scheduler_state_dict" in checkpoint
        ):
            self.scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        self.history = checkpoint["history"]

        self.best_validation_accuracy = checkpoint[
            "best_validation_accuracy"
        ]

        print(f"Checkpoint loaded from {path}")

        return checkpoint["epoch"]


    def close(self):

        self.writer.close()