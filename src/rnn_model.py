"""RNN/LSTM baseline model for sentence classification."""

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """Bidirectional LSTM model for text classification.

    Architecture:
        Embedding -> BiLSTM -> Dropout -> Linear -> Softmax
    """

    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        hidden_dim=256,
        num_layers=2,
        num_classes=5,
        dropout=0.3,
        pad_idx=0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len).

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        lstm_out, (hidden, _) = self.lstm(embedded)

        # Concatenate final forward and backward hidden states
        hidden_fwd = hidden[-2]  # (batch, hidden_dim)
        hidden_bwd = hidden[-1]  # (batch, hidden_dim)
        hidden_cat = torch.cat([hidden_fwd, hidden_bwd], dim=1)

        out = self.dropout(hidden_cat)
        logits = self.fc(out)
        return logits
