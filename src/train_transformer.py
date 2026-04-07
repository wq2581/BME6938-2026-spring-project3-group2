"""Training script for BioBERT transformer model."""

import os
import json
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.data_loader import NUM_LABELS, ID_TO_LABEL
from src.transformer_model import MODEL_NAME, tokenize_dataset
from src.evaluate import (
    compute_metrics, print_classification_report,
    plot_confusion_matrix, plot_training_curves,
)


# Hyperparameters
MAX_LENGTH = 128
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
MODEL_DIR = os.path.join(SAVE_DIR, "best_transformer_model")


def compute_metrics_hf(eval_pred):
    """Compute metrics for HuggingFace Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train():
    """Main training function."""
    print(f"Using device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load data
    print("Loading PubMed RCT 20k dataset...")
    dataset = load_dataset("armanc/pubmed-rct20k")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenize datasets
    print("Tokenizing datasets...")
    train_dataset = tokenize_dataset(dataset["train"], tokenizer, MAX_LENGTH)
    val_dataset = tokenize_dataset(dataset["validation"], tokenizer, MAX_LENGTH)
    test_dataset = tokenize_dataset(dataset["test"], tokenizer, MAX_LENGTH)

    # Load model
    print("Loading BioBERT model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID_TO_LABEL,
        label2id={v: k for k, v in ID_TO_LABEL.items()},
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(SAVE_DIR, "transformer_checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=os.path.join(SAVE_DIR, "logs"),
        logging_steps=100,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_hf,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Train
    print("Starting training...")
    train_result = trainer.train()
    print(f"Training completed. Total steps: {train_result.global_step}")

    # Save best model
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Best model saved to {MODEL_DIR}/")

    # Test evaluation
    print("\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)
    test_results = trainer.predict(test_dataset)
    test_preds = np.argmax(test_results.predictions, axis=-1)
    test_labels = test_results.label_ids

    print(f"Test Accuracy: {test_results.metrics['test_accuracy']:.4f}")
    print(f"Test Macro F1: {test_results.metrics['test_f1']:.4f}")
    print_classification_report(test_labels, test_preds)

    # Save figures
    print("Saving figures...")
    plot_confusion_matrix(
        test_labels, test_preds,
        title="BioBERT - Test Confusion Matrix",
        save_path=os.path.join(SAVE_DIR, "transformer_confusion_matrix.png"),
    )

    # Extract per-epoch metrics from trainer log for training curves
    log_history = trainer.state.log_history
    train_losses = [e["loss"] for e in log_history if "loss" in e and "eval_loss" not in e]
    eval_entries = [e for e in log_history if "eval_loss" in e]
    val_losses = [e["eval_loss"] for e in eval_entries]
    val_accs = [e["eval_accuracy"] for e in eval_entries]
    # Sample train losses at epoch boundaries to match val length
    if train_losses and val_losses:
        step = max(1, len(train_losses) // len(val_losses))
        epoch_train_losses = train_losses[step - 1::step][:len(val_losses)]
        plot_training_curves(
            epoch_train_losses, val_losses,
            train_accs=None, val_accs=val_accs,
            title="BioBERT",
            save_path=os.path.join(SAVE_DIR, "transformer_training_curves.png"),
        )

    # Save metrics
    history = {
        "train_loss": train_result.training_loss,
        "test_metrics": {
            k: float(v) for k, v in test_results.metrics.items()
        },
    }
    with open(os.path.join(SAVE_DIR, "transformer_test_metrics.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Save training log
    with open(os.path.join(SAVE_DIR, "transformer_history.json"), "w") as f:
        json.dump(log_history, f, indent=2, default=float)

    print(f"\nResults saved to {SAVE_DIR}/")
    return model, history


if __name__ == "__main__":
    train()
