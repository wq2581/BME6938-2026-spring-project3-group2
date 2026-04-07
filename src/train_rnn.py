"""Training script for RNN/LSTM baseline model."""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_loader import (
    load_pubmed_rct,
    PubMedRCTDataset,
    build_vocab,
    encode_text,
    NUM_LABELS,
)
from src.rnn_model import LSTMClassifier
from src.evaluate import (
    compute_metrics, print_classification_report,
    plot_confusion_matrix, plot_training_curves,
)


# Hyperparameters
BATCH_SIZE = 128
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
MAX_LEN = 128
MIN_FREQ = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def collate_fn(batch, vocab, max_len):
    """Convert a batch of (text, label) into tensors."""
    texts, labels = zip(*batch)
    input_ids = [encode_text(t, vocab, max_len) for t in texts]
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def evaluate_model(model, dataloader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    metrics = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics, all_labels, all_preds


def train():
    """Main training loop."""
    print(f"Using device: {DEVICE}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load data
    print("Loading PubMed RCT 20k dataset...")
    train_split, val_split, test_split = load_pubmed_rct()

    train_dataset = PubMedRCTDataset(train_split, max_len=MAX_LEN)
    val_dataset = PubMedRCTDataset(val_split, max_len=MAX_LEN)
    test_dataset = PubMedRCTDataset(test_split, max_len=MAX_LEN)

    # Build vocabulary
    print("Building vocabulary...")
    vocab = build_vocab(train_dataset, min_freq=MIN_FREQ)
    print(f"Vocabulary size: {len(vocab)}")

    # Save vocab
    with open(os.path.join(SAVE_DIR, "vocab.json"), "w") as f:
        json.dump(vocab, f)

    # Create data loaders
    collate = lambda batch: collate_fn(batch, vocab, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate, num_workers=2)

    # Initialize model
    model = LSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=NUM_LABELS,
        dropout=DROPOUT,
    ).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    # Training loop
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_f1 = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = total_loss / len(all_labels)
        train_metrics = compute_metrics(all_labels, all_preds)

        # Validation
        val_loss, val_metrics, _, _ = evaluate_model(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss={train_loss:.4f}, Train Acc={train_metrics['accuracy']:.4f} | "
            f"Val Loss={val_loss:.4f}, Val Acc={val_metrics['accuracy']:.4f}, "
            f"Val F1={val_metrics['macro_f1']:.4f}"
        )

        # Save best model
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_rnn_model.pt"))
            print(f"  -> Saved best model (F1={best_val_f1:.4f})")

    # Test evaluation
    print("\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_rnn_model.pt"),
                                     weights_only=True))
    test_loss, test_metrics, test_labels, test_preds = evaluate_model(
        model, test_loader, criterion, DEVICE
    )
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
    print_classification_report(test_labels, test_preds)

    # Save figures
    print("Saving figures...")
    plot_confusion_matrix(
        test_labels, test_preds,
        title="RNN/LSTM - Test Confusion Matrix",
        save_path=os.path.join(SAVE_DIR, "rnn_confusion_matrix.png"),
    )
    plot_training_curves(
        history["train_loss"], history["val_loss"],
        history["train_acc"], history["val_acc"],
        title="RNN/LSTM",
        save_path=os.path.join(SAVE_DIR, "rnn_training_curves.png"),
    )

    # Save results
    with open(os.path.join(SAVE_DIR, "rnn_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(SAVE_DIR, "rnn_test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2, default=float)

    print(f"\nResults saved to {SAVE_DIR}/")
    return model, history, test_metrics


if __name__ == "__main__":
    train()
