import pytorch_lightning as pl
from torch import nn
import torch

class SmallTextGeneratorModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch['input_ids']
        y = batch['labels']
        y_hat = self(x)
        loss = self.loss(y_hat.view(-1, y_hat.size(-1)), y.view(-1))
        self.log("train_perplexity", torch.exp(loss))
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['input_ids']
        y = batch['labels']
        y_hat = self(x)
        loss = self.loss(y_hat.view(-1, y_hat.size(-1)), y.view(-1))
        self.log("val_perplexity", torch.exp(loss))
        accuracy = (y_hat.argmax(dim=-1) == y).float().mean()
        self.log("val_accuracy", accuracy)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch['input_ids']
        y = batch['labels']
        y_hat = self(x)
        loss = self.loss(y_hat.view(-1, y_hat.size(-1)), y.view(-1))
        self.log("test_perplexity", torch.exp(loss))
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)


    