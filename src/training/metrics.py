import torch


def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute classification accuracy for one batch.

    Parameters
    ----------
    outputs : torch.Tensor
        Raw logits from the model.

    labels : torch.Tensor
        Ground-truth labels.

    Returns
    -------
    float
        Batch accuracy.
    """

    predictions = outputs.argmax(dim=1)
    correct = (predictions == labels).sum().item()

    return correct / labels.size(0)