import torch
from torch import nn


class SmallTextGenerator(nn.Module):
    def __init__(self, d_model, vocab_size, num_layers, num_heads, dim_feedforward, dropout):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, num_heads, dim_feedforward, dropout),
            num_layers
        )
        self.linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_decoder(x, x)
        x = self.linear(x)
        return x
