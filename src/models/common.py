"""
common.py

Provides the BaseCNN class that all models in this framework should inherit from.
This adds common utilities like saving, loading, parameter counting, and freezing.
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class BaseCNN(nn.Module):
    """
    Base class for all Convolutional Neural Networks in the project.
    Provides standard utility methods for research workflows.
    """

    def __init__(self):
        super().__init__()

    def freeze(self):
        """
        Freezes all parameters in the model (requires_grad = False).
        """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """
        Unfreezes all parameters in the model (requires_grad = True).
        """
        for param in self.parameters():
            param.requires_grad = True

    def num_parameters(self, trainable_only: bool = True) -> int:
        """
        Returns the number of parameters in the model.
        
        Parameters
        ----------
        trainable_only : bool
            If True, only counts parameters with requires_grad=True.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

    def save(self, path: str, extra_metadata: Dict[str, Any] = None):
        """
        Saves the model state dictionary and optional metadata.
        
        Parameters
        ----------
        path : str
            Path to save the .pth file.
        extra_metadata : dict, optional
            Additional metadata to store alongside the model.
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
        }
        if extra_metadata:
            checkpoint.update(extra_metadata)
            
        torch.save(checkpoint, path)

    def load(self, path: str, device: torch.device = None) -> Dict[str, Any]:
        """
        Loads the model state dictionary.
        
        Parameters
        ----------
        path : str
            Path to load the .pth file from.
        device : torch.device, optional
            Device to load the tensors to.
            
        Returns
        -------
        dict
            The loaded checkpoint (including any metadata).
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        checkpoint = torch.load(path, map_location=device)
        
        if "model_state_dict" in checkpoint:
            self.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.load_state_dict(checkpoint)
            
        return checkpoint

    def summary(self):
        """
        Prints a high-level summary of the model structure and parameters.
        """
        print("=" * 60)
        print(f"Model Summary: {self.__class__.__name__}")
        print("-" * 60)
        print(self)
        print("-" * 60)
        print(f"Total Parameters: {self.num_parameters(trainable_only=False):,}")
        print(f"Trainable Parameters: {self.num_parameters(trainable_only=True):,}")
        print("=" * 60)
