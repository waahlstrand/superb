import torch
from torch.nn import functional as F
import torch.nn as nn
import pytorch_lightning as pl

class SuperbModel(pl.LightningModule):

    def __init__(self, height: int, width: int, channels: int, num_classes: int):

        self.height = height
        self.width = width
        self.channels = channels
        self.num_classes = num_classes


        super().__init__()
        self.model = nn.Sequential(

        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    

model = SuperbModel(512, 512, 3, 2)
trainer = pl.Trainer(gpus=1, max_epochs=10, precision=16)
trainer.fit(model)