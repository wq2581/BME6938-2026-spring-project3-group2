"""Data loading utilities for PubMed RCT 20k dataset."""

import re
from datasets import load_dataset
from torch.utils.data import Dataset


LABEL_MAP = {
    "BACKGROUND": 0,
    "OBJECTIVE": 1,
    "METHODS": 2,
    "RESULTS": 3,
    "CONCLUSIONS": 4,
}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = len(LABEL_MAP)


def load_pubmed_rct():
    """Load PubMed RCT 20k dataset from Hugging Face.

    Returns train, validation, and test splits as HuggingFace datasets.
    """
    dataset = load_dataset("armanc/pubmed-rct20k")
    return dataset["train"], dataset["validation"], dataset["test"]


def preprocess_text(text: str) -> str:
    """Clean and normalize biomedical text."""
    text = text.lower()
    text = re.sub(r"\d+", " NUM ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class PubMedRCTDataset(Dataset):
    """PyTorch Dataset for PubMed RCT sentence classification."""

    def __init__(self, hf_dataset, vocab=None, max_len=128, preprocess=True):
        self.texts = []
        self.labels = []
        self.max_len = max_len
        self.vocab = vocab

        for example in hf_dataset:
            text = example["text"]
            if preprocess:
                text = preprocess_text(text)
            self.texts.append(text)
            label = example["label"]
            if isinstance(label, str):
                self.labels.append(LABEL_MAP[label.upper()])
            else:
                self.labels.append(label)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def build_vocab(dataset, min_freq=2):
    """Build vocabulary from dataset texts.

    Args:
        dataset: PubMedRCTDataset instance.
        min_freq: Minimum word frequency to include.

    Returns:
        Dictionary mapping words to indices.
    """
    word_counts = {}
    for text, _ in dataset:
        for word in text.split():
            word_counts[word] = word_counts.get(word, 0) + 1

    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for word, count in sorted(word_counts.items()):
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab


def encode_text(text, vocab, max_len):
    """Convert text to a list of vocabulary indices."""
    tokens = text.split()[:max_len]
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    # Pad to max_len
    ids += [vocab["<PAD>"]] * (max_len - len(ids))
    return ids
