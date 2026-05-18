"""Recurrent classifier for MFCC sequence input (batch, T, n_mfcc).
Supports RNN, LSTM, GRU — unidirectional or bidirectional.
"""
import torch
import torch.nn as nn


class SequenceEmotionClassifier(nn.Module):
    """
    Input:  (batch, T, input_size)
    Output: (batch, n_classes)
    """

    SUPPORTED = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}

    def __init__(
        self,
        model_type: str = "lstm",
        input_size: int = 40,
        hidden_size: int = 128,
        num_layers: int = 2,
        n_classes: int = 6,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        if model_type not in self.SUPPORTED:
            raise ValueError(f"model_type must be one of {list(self.SUPPORTED)}")

        self.rnn = self.SUPPORTED[model_type](
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        rnn_out = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(rnn_out, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.classifier(out[:, -1, :])   # last time-step
