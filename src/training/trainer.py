import torch
import torch.optim as optim

from src.training.losses import get_loss


class Trainer:

    """
    Generic Trainer for image classification models.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        epochs=20,
        learning_rate=1e-4,
        device=None,
        loss_name="cross_entropy",
    ):

        self.model = model

        self.train_loader = train_loader

        self.val_loader = val_loader

        self.epochs = epochs

        self.device = (
            device
            if device is not None
            else torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.model.to(self.device)

        self.criterion = get_loss(loss_name)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
        )

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def train_one_epoch(self):
        """
        Implement after model creation.
        """
        raise NotImplementedError

    def validate(self):
        """
        Implement after model creation.
        """
        raise NotImplementedError

    def fit(self):
        """
        Full training loop.

        Will be implemented after LeNet.
        """
        raise NotImplementedError