"""Transformer-based model for sentence classification using BioBERT."""

from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"


def load_tokenizer(model_name=MODEL_NAME):
    """Load the BioBERT tokenizer."""
    return AutoTokenizer.from_pretrained(model_name)


def load_model(model_name=MODEL_NAME, num_labels=5):
    """Load BioBERT for sequence classification."""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    return model


def tokenize_dataset(dataset, tokenizer, max_length=128):
    """Tokenize a HuggingFace dataset for transformer input.

    Args:
        dataset: HuggingFace dataset with 'text' and 'label' fields.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length.

    Returns:
        Tokenized dataset ready for training.
    """
    from src.data_loader import LABEL_MAP

    def tokenize_fn(examples):
        tokens = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        # Convert string labels to integers
        labels = examples["label"]
        tokens["labels"] = [
            LABEL_MAP[l.upper()] if isinstance(l, str) else l for l in labels
        ]
        return tokens

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.remove_columns(
        [c for c in tokenized.column_names if c not in ["input_ids", "attention_mask", "labels"]]
    )
    tokenized.set_format("torch")
    return tokenized
