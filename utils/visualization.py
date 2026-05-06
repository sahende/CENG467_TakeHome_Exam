"""
Visualization utilities for the CENG467 NLP project.
Provides plotting functions for results analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
import torch

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def set_plotting_style(figsize: Tuple[int, int] = (10, 6),
                       font_scale: float = 1.2):
    """
    Set global plotting style.
    
    Args:
        figsize: Default figure size
        font_scale: Font scale factor
    """
    plt.rcParams['figure.figsize'] = figsize
    sns.set_context("notebook", font_scale=font_scale)


def plot_training_history(history: Dict, title: str = "Training History",
                          save_path: Optional[str] = None):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Dictionary with training history
        title: Plot title
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    if 'train_loss' in history and history['train_loss']:
        axes[0].plot(history['epoch'], history['train_loss'], 
                    label='Train Loss', marker='o', markersize=4)
    if 'val_loss' in history and history['val_loss']:
        axes[0].plot(history['epoch'], history['val_loss'], 
                    label='Validation Loss', marker='s', markersize=4)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    if 'train_acc' in history and history['train_acc']:
        axes[1].plot(history['epoch'], history['train_acc'], 
                    label='Train Accuracy', marker='o', markersize=4)
    if 'val_acc' in history and history['val_acc']:
        axes[1].plot(history['epoch'], history['val_acc'], 
                    label='Validation Accuracy', marker='s', markersize=4)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix_heatmap(y_true: np.ndarray, y_pred: np.ndarray,
                                   class_names: List[str] = None,
                                   normalize: bool = True,
                                   title: str = "Confusion Matrix",
                                   save_path: Optional[str] = None):
    """
    Plot confusion matrix as heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: Whether to normalize values
        title: Plot title
        save_path: Path to save figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(8, 6))
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(cm.shape[0])]
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion' if normalize else 'Count'})
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_model_comparison(models: List[str], metrics: Dict[str, List[float]],
                          metric_name: str = "F1 Score",
                          title: str = "Model Comparison",
                          save_path: Optional[str] = None,
                          color_palette: str = "viridis"):
    """
    Create bar plot comparing multiple models on a metric.
    
    Args:
        models: List of model names
        metrics: Dictionary with metric values for each model
        metric_name: Name of the metric
        title: Plot title
        save_path: Path to save figure
        color_palette: Color palette name
    """
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.8 / len(metrics)
    
    colors = sns.color_palette(color_palette, len(metrics))
    
    for i, (metric_key, values) in enumerate(metrics.items()):
        bars = plt.bar(x + i * width, values, width, 
                      label=metric_key, color=colors[i], alpha=0.8)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Models')
    plt.ylabel(metric_name)
    plt.title(title)
    plt.xticks(x + width * (len(metrics) - 1) / 2, models)
    plt.legend()
    plt.ylim(0, max(max(v) for v in metrics.values()) * 1.15)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_ner_error_analysis(error_types: Dict[str, int],
                            title: str = "NER Error Analysis",
                            save_path: Optional[str] = None):
    """
    Plot NER error distribution.
    
    Args:
        error_types: Dictionary mapping error types to counts
        title: Plot title
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart
    axes[0].pie(error_types.values(), labels=error_types.keys(),
                autopct='%1.1f%%', startangle=90,
                colors=sns.color_palette("pastel", len(error_types)))
    axes[0].set_title('Error Distribution')
    
    # Bar chart
    sorted_items = sorted(error_types.items(), key=lambda x: x[1], reverse=True)
    keys, values = zip(*sorted_items)
    
    bars = axes[1].bar(range(len(keys)), values, color=sns.color_palette("husl", len(keys)))
    axes[1].set_xticks(range(len(keys)))
    axes[1].set_xticklabels(keys, rotation=45, ha='right')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Error Count by Type')
    
    # Add value labels
    for bar, val in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(val), ha='center', va='bottom')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_rouge_comparison(extractive_scores: Dict[str, float],
                          abstractive_scores: Dict[str, float],
                          save_path: Optional[str] = None):
    """
    Compare ROUGE scores between extractive and abstractive methods.
    
    Args:
        extractive_scores: ROUGE scores for extractive method
        abstractive_scores: ROUGE scores for abstractive method
        save_path: Path to save figure
    """
    metrics = list(extractive_scores.keys())
    ext_values = [extractive_scores[m] for m in metrics]
    abs_values = [abstractive_scores[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, ext_values, width, label='Extractive',
                   color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, abs_values, width, label='Abstractive',
                   color='#3498db', alpha=0.8)
    
    ax.set_xlabel('ROUGE Metric')
    ax.set_ylabel('F1 Score')
    ax.set_title('ROUGE Score Comparison: Extractive vs Abstractive')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylim(0, max(max(ext_values), max(abs_values)) * 1.15)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_attention_weights(attention_weights: np.ndarray,
                           source_tokens: List[str],
                           target_tokens: List[str],
                           title: str = "Attention Weights",
                           save_path: Optional[str] = None):
    """
    Visualize attention weights between source and target sequences.
    
    Args:
        attention_weights: Attention matrix [target_len, source_len]
        source_tokens: Source sequence tokens
        target_tokens: Target sequence tokens
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(12, 8))
    
    sns.heatmap(attention_weights, 
                xticklabels=source_tokens,
                yticklabels=target_tokens,
                cmap='YlOrRd',
                cbar_kws={'label': 'Attention Weight'})
    
    plt.xlabel('Source Tokens')
    plt.ylabel('Target Tokens')
    plt.title(title)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_perplexity_comparison(models: List[str], 
                               perplexities: List[float],
                               title: str = "Perplexity Comparison",
                               save_path: Optional[str] = None):
    """
    Compare perplexity scores across language models.
    
    Args:
        models: Model names
        perplexities: Perplexity scores
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(8, 5))
    
    colors = sns.color_palette("rocket", len(models))
    bars = plt.bar(models, perplexities, color=colors, alpha=0.8)
    
    # Add value labels
    for bar, ppl in zip(bars, perplexities):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{ppl:.1f}', ha='center', va='bottom', fontsize=11,
                fontweight='bold')
    
    plt.xlabel('Model')
    plt.ylabel('Perplexity')
    plt.title(title)
    
    # Add horizontal line for reference
    if perplexities:
        plt.axhline(y=min(perplexities), color='green', linestyle='--', 
                   alpha=0.5, label=f'Best: {min(perplexities):.1f}')
        plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_text_length_distribution(texts: List[str], 
                                  labels: Optional[List[int]] = None,
                                  title: str = "Text Length Distribution",
                                  save_path: Optional[str] = None):
    """
    Plot histogram of text lengths, optionally colored by label.
    
    Args:
        texts: List of text strings
        labels: Optional labels for color coding
        title: Plot title
        save_path: Path to save figure
    """
    lengths = [len(text.split()) for text in texts]
    
    plt.figure(figsize=(10, 6))
    
    if labels is not None:
        for label in set(labels):
            label_lengths = [l for l, lab in zip(lengths, labels) if lab == label]
            plt.hist(label_lengths, bins=30, alpha=0.6, 
                    label=f'Class {label}', density=True)
        plt.legend()
    else:
        plt.hist(lengths, bins=30, alpha=0.7, edgecolor='black')
    
    plt.xlabel('Number of Words')
    plt.ylabel('Density')
    plt.title(title)
    
    # Add statistics
    mean_len = np.mean(lengths)
    median_len = np.median(lengths)
    plt.axvline(mean_len, color='red', linestyle='--', 
               label=f'Mean: {mean_len:.1f}')
    plt.axvline(median_len, color='green', linestyle='--', 
               label=f'Median: {median_len:.1f}')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def save_all_figures(figures_dir: str = "figures"):
    """
    Create directory for saving figures.
    
    Args:
        figures_dir: Directory path for figures
        
    Returns:
        Path to figures directory
    """
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


if __name__ == "__main__":
    # Example usage
    print("Visualization utilities ready.")
    
    # Example: Plot model comparison
    plot_model_comparison(
        models=['TF-IDF', 'BiLSTM', 'BERT'],
        metrics={'Accuracy': [0.85, 0.88, 0.92], 'F1 Score': [0.84, 0.87, 0.91]},
        metric_name="Score",
        title="Model Comparison Example",
        save_path="figures/model_comparison_example.png"
    )