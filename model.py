import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


class CNNLSTM(nn.Module):
    def __init__(
        self,
        num_classes=6,
        hidden_size=256,
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()

        # Same MobileNetV3-Small architecture used in training.
        # weights=None is intentional: the trained CNN weights
        # are loaded from the checkpoint below.
        self.cnn = mobilenet_v3_small(weights=None)
        self.cnn.classifier = nn.Identity()

        self.feature_size = 576

        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, frames, channels, height, width = x.shape

        x = x.view(
            batch_size * frames,
            channels,
            height,
            width
        )

        x = self.cnn(x)

        x = x.view(
            batch_size,
            frames,
            self.feature_size
        )

        lstm_out, _ = self.lstm(x)

        x = lstm_out[:, -1, :]
        x = self.dropout(x)

        return self.fc(x)
