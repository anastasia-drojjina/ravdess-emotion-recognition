"""CNN classifier for mel-spectrogram input (1, n_mels, T)."""
import torch
import torch.nn as nn


def _conv_block(in_ch, out_ch, dropout: float = 0.0):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Dropout2d(dropout),
    )


class CNNEmotionClassifier(nn.Module):
    """
    4-block CNN with global average pooling.
    Input:  (batch, 1, n_mels, T)
    Output: (batch, n_classes)
    """

    def __init__(self, n_classes: int = 6, n_mels: int = 128, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(1,   32,  dropout / 2),
            _conv_block(32,  64,  dropout / 2),
            _conv_block(64,  128, dropout / 2),
            _conv_block(128, 256, dropout / 2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
