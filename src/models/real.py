from __future__ import annotations

from src.models.base import BaseModel, nn, torch_available


class RealModel(BaseModel):
    def __init__(self, input_channels: int = 1, num_classes: int = 2) -> None:
        super().__init__(input_channels=input_channels, num_classes=num_classes)
        if not torch_available():
            return
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(32, num_classes)

    def forward(self, inputs):
        features = self.encoder(inputs)
        return self.head(features.flatten(start_dim=1))
