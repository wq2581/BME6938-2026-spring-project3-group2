"""Evaluation utilities for model performance analysis."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from src.data_loader import ID_TO_LABEL


def compute_metrics(y_true, y_pred, label_names=None):
    """Compute classification metrics.

    Returns:
        Dictionary with accuracy, per-class precision/recall/F1, and macro averages.
    """
    if label_names is None:
        label_names = [ID_TO_LABEL[i] for i in range(len(ID_TO_LABEL))]

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(label_names))
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )

    metrics = {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": {},
    }
    for i, name in enumerate(label_names):
        metrics["per_class"][name] = {
            "precision": precision[i],
            "recall": recall[i],
            "f1": f1[i],
            "support": int(support[i]),
        }
    return metrics


def print_classification_report(y_true, y_pred, label_names=None):
    """Print sklearn classification report."""
    if label_names is None:
        label_names = [ID_TO_LABEL[i] for i in range(len(ID_TO_LABEL))]
    print(classification_report(y_true, y_pred, target_names=label_names, digits=4))


def plot_confusion_matrix(y_true, y_pred, label_names=None, title="Confusion Matrix",
                          save_path=None):
    """Plot and optionally save a confusion matrix heatmap."""
    if label_names is None:
        label_names = [ID_TO_LABEL[i] for i in range(len(ID_TO_LABEL))]

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_training_curves(train_losses, val_losses, train_accs=None, val_accs=None,
                         title="Training Curves", save_path=None):
    """Plot training and validation loss/accuracy curves."""
    fig, axes = plt.subplots(1, 2 if train_accs else 1, figsize=(12, 5))
    if train_accs is None:
        axes = [axes]

    axes[0].plot(train_losses, label="Train Loss")
    axes[0].plot(val_losses, label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} - Loss")
    axes[0].legend()

    if train_accs and val_accs:
        axes[1].plot(train_accs, label="Train Acc")
        axes[1].plot(val_accs, label="Val Acc")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title(f"{title} - Accuracy")
        axes[1].legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig
