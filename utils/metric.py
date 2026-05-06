"""
Shared utility functions for metrics calculation across all questions.
Provides common evaluation metrics used throughout the project.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os


class MetricsTracker:
    """
    Tracks and stores metrics during training.
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'epoch': []
        }
    
    def update(self, epoch: int, train_loss: float = None,
               val_loss: float = None, train_acc: float = None,
               val_acc: float = None):
        """
        Update metrics for an epoch.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            val_loss: Validation loss
            train_acc: Training accuracy
            val_acc: Validation accuracy
        """
        self.history['epoch'].append(epoch)
        
        if train_loss is not None:
            self.history['train_loss'].append(train_loss)
        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
        if train_acc is not None:
            self.history['train_acc'].append(train_acc)
        if val_acc is not None:
            self.history['val_acc'].append(val_acc)
    
    def get_best_epoch(self, metric: str = 'val_acc') -> int:
        """
        Get epoch with best metric value.
        
        Args:
            metric: Metric to use ('val_acc' or 'val_loss')
            
        Returns:
            Best epoch number
        """
        if metric not in self.history or len(self.history[metric]) == 0:
            return -1
        
        if 'loss' in metric:
            return self.history['epoch'][np.argmin(self.history[metric])]
        else:
            return self.history['epoch'][np.argmax(self.history[metric])]
    
    def to_dict(self) -> Dict:
        """Convert history to dictionary."""
        return self.history
    
    def save(self, filepath: str):
        """Save metrics history to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=4)
    
    def load(self, filepath: str):
        """Load metrics history from JSON."""
        with open(filepath, 'r') as f:
            self.history = json.load(f)


def compute_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                               average: str = 'macro') -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method ('macro', 'micro', 'weighted')
        
    Returns:
        Dictionary with accuracy, precision, recall, F1
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    # Per-class metrics for binary classification
    if len(np.unique(y_true)) == 2:
        metrics['precision_per_class'] = precision_score(
            y_true, y_pred, average=None, zero_division=0
        ).tolist()
        metrics['recall_per_class'] = recall_score(
            y_true, y_pred, average=None, zero_division=0
        ).tolist()
        metrics['f1_per_class'] = f1_score(
            y_true, y_pred, average=None, zero_division=0
        ).tolist()
    
    return metrics


def format_metrics(metrics_dict: Dict, decimals: int = 4) -> str:
    """
    Format metrics dictionary into readable string.
    
    Args:
        metrics_dict: Dictionary of metrics
        decimals: Number of decimal places
        
    Returns:
        Formatted string
    """
    lines = []
    for key, value in metrics_dict.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.{decimals}f}")
        elif isinstance(value, list):
            formatted_vals = [f"{v:.{decimals}f}" for v in value]
            lines.append(f"{key}: [{', '.join(formatted_vals)}]")
        else:
            lines.append(f"{key}: {value}")
    
    return "\n".join(lines)


def calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from cross-entropy loss.
    
    Args:
        loss: Cross-entropy loss value
        
    Returns:
        Perplexity score
    """
    return np.exp(loss)


def sequence_accuracy(y_true: List[List[str]], 
                      y_pred: List[List[str]]) -> float:
    """
    Calculate sequence-level accuracy for NER tasks.
    
    Args:
        y_true: List of true label sequences
        y_pred: List of predicted label sequences
        
    Returns:
        Sequence accuracy
    """
    correct = 0
    total = 0
    
    for true_seq, pred_seq in zip(y_true, y_pred):
        for t, p in zip(true_seq, pred_seq):
            if t == p:
                correct += 1
            total += 1
    
    return correct / total if total > 0 else 0.0


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Count trainable and total parameters in a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (trainable_params, total_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    
    return trainable_params, total_params


def format_param_count(trainable: int, total: int) -> str:
    """
    Format parameter count in human-readable format.
    
    Args:
        trainable: Number of trainable parameters
        total: Total number of parameters
        
    Returns:
        Formatted string
    """
    if total >= 1e6:
        return (f"Trainable params: {trainable/1e6:.1f}M | "
                f"Total params: {total/1e6:.1f}M")
    elif total >= 1e3:
        return (f"Trainable params: {trainable/1e3:.1f}K | "
                f"Total params: {total/1e3:.1f}K")
    else:
        return f"Trainable params: {trainable} | Total params: {total}"


def early_stopping_check(val_losses: List[float], patience: int = 3,
                         min_delta: float = 0.001) -> bool:
    """
    Check if early stopping criterion is met.
    
    Args:
        val_losses: List of validation losses
        patience: Number of epochs to wait
        min_delta: Minimum improvement threshold
        
    Returns:
        True if training should stop
    """
    if len(val_losses) < patience + 1:
        return False
    
    best_loss = min(val_losses[:-patience])
    recent_losses = val_losses[-patience:]
    
    # Check if all recent losses are worse than best
    return all(loss > best_loss + min_delta for loss in recent_losses)