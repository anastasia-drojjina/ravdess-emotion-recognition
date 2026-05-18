"""MLP baseline: collapses the time dimension with mean-pooling, then classifies."""
import torch
import torch.nn as nn


class MLPEmotionClassifier(nn.Module):
    """
    Input:  (batch, T, input_size)  or  (batch, input_size)
    Output: (batch, n_classes)
    """

    def __init__(
        self,
        input_size: int = 40,
        hidden_sizes: list = None,
        n_classes: int = 6,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128]

        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.mean(dim=1)   # (batch, T, F) → (batch, F)
        return self.net(x)
