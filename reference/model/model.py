import torch
from torch import nn


class SmallTextGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.embedding = nn.Embedding(1000, 64)
        self.transformer_decoder = nn.Transformer(
            d_model=64,
            nhead=8,
            num_encoder_layers=0,
            num_decoder_layers=6,
            dim_feedforward=256,
            dropout=0.1,
            activation="relu",
        )
        self.linear = nn.Linear(64, 1000)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_decoder(x, x)
        x = self.linear(x)
        return x
