"""Trainer for binary segmentation, refactored from the v1 EuroSAT Trainer
(legacy/eurosat/src/training.py): keeps AMP, LR schedulers, early stopping, best/last
checkpoints, TensorBoard and history.json; replaces accuracy with a pluggable val metric."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from . import metrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def masked_bce_dice(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """BCE + soft Dice over valid pixels only."""
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp(min=1)
    p = torch.sigmoid(logits) * valid
    t = target * valid
    dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)
    return bce + dice


@torch.no_grad()
def predict_logits(model: torch.nn.Module, x: torch.Tensor, device: torch.device, batch: int = 16) -> np.ndarray:
    """(N,C,H,W) float tensor -> (N,H,W) float32 logits."""
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        out.append(model(x[i:i + batch].to(device)).float().cpu()[:, 0])
    return torch.cat(out).numpy()


def iou_metric(model, loader, device) -> dict[str, float]:
    c = []
    model.eval()
    with torch.no_grad():
        for b in loader:
            logits = model(b["image"].to(device)).float().cpu()
            c.append(metrics.confusion((logits > 0).numpy(), b["mask"].numpy() > 0.5, b["valid"].numpy() > 0.5))
    return metrics.scores(metrics.add(c))


class Trainer:
    def __init__(self, model: torch.nn.Module, train_loader, val_loader, optimizer: torch.optim.Optimizer,
                 run_dir: str | Path, epochs: int = 50, scheduler: str = "cosine",
                 loss_fn: Callable = masked_bce_dice, metric_fn: Callable = iou_metric,
                 monitor: str = "iou", patience: int = 15, use_amp: bool = True,
                 device: torch.device | None = None, max_batches: int | None = None):
        self.model, self.train_loader, self.val_loader = model, train_loader, val_loader
        self.optimizer, self.loss_fn, self.metric_fn = optimizer, loss_fn, metric_fn
        self.epochs, self.monitor, self.patience, self.max_batches = epochs, monitor, patience, max_batches
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                          if scheduler == "cosine" else None)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.run_dir / "tb"))
        self.model.to(self.device)
        self.history: list[dict] = []
        self.best, self.best_epoch = -np.inf, -1

    def train_epoch(self) -> float:
        self.model.train()
        total, n = 0.0, 0
        for i, b in enumerate(self.train_loader):
            if self.max_batches and i >= self.max_batches:
                break
            x, y, v = (b[k].to(self.device) for k in ("image", "mask", "valid"))
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                loss = self.loss_fn(self.model(x).float(), y, v)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total, n = total + loss.item() * len(x), n + len(x)
        return total / max(n, 1)

    def save(self, name: str, epoch: int) -> None:
        torch.save({"model_state_dict": self.model.state_dict(), "epoch": epoch}, self.run_dir / name)

    def fit(self) -> list[dict]:
        bad = 0
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch()
            val = self.metric_fn(self.model, self.val_loader, self.device)
            lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler:
                self.scheduler.step()
            row = {"epoch": epoch, "train_loss": train_loss, "lr": lr, "time_s": time.time() - t0,
                   **{f"val_{k}": v for k, v in val.items()}}
            self.history.append(row)
            self.writer.add_scalar("loss/train", train_loss, epoch)
            for k, v in val.items():
                self.writer.add_scalar(f"val/{k}", v, epoch)
            if val[self.monitor] > self.best:
                self.best, self.best_epoch, bad = val[self.monitor], epoch, 0
                self.save("best.pt", epoch)
            else:
                bad += 1
            self.save("last.pt", epoch)
            print(f"epoch {epoch:3d} loss {train_loss:.4f} val_{self.monitor} {val[self.monitor]:.4f} "
                  f"(best {self.best:.4f} @ {self.best_epoch}) {row['time_s']:.1f}s", flush=True)
            if bad >= self.patience:
                print(f"early stop at epoch {epoch}")
                break
        (self.run_dir / "history.json").write_text(json.dumps(self.history, indent=1))
        self.writer.close()
        return self.history

    def load_best(self) -> torch.nn.Module:
        ckpt = torch.load(self.run_dir / "best.pt", map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        return self.model
