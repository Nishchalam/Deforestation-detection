import torch.nn as nn


def get_loss(loss_name="cross_entropy"):
    """
    Factory function for loss functions.
    """

    loss_name = loss_name.lower()

    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()

    raise ValueError(f"Unknown loss function: {loss_name}")