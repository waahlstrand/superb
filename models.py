import torch
from torch.nn import functional as F
import torch.nn as nn
from torchvision.models import resnet18, resnet34
import pytorch_lightning as pl
from preprocessing.images import PADDING_SHAPE
import warnings
import torchmetrics
from typing import *

warnings.filterwarnings("ignore")

class SuperbModel(pl.LightningModule):

    def __init__(self, n_channels: int, 
                       n_classes: int, 
                       lr: float = 1e-3,
                       height: int = PADDING_SHAPE[0],
                       width: int = PADDING_SHAPE[1], ):

        self.height = height # 700
        self.width = width # 1800
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.lr = lr
        self.kernel = (2, 3)

        super().__init__()

        self.metrics = self.set_metrics("binary", n_classes) if n_classes == 1 else self.set_metrics("multiclass", n_classes)

        self.pre_conv = nn.Conv2d(in_channels=n_channels, out_channels=3, kernel_size=self.kernel, stride=1, padding=0)
        self.resnet = resnet34(pretrained=False)
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1000, n_classes),
            nn.Sigmoid()
        )

    def set_metrics(task, n_classes: int) -> List[torchmetrics.Metric]:

        metrics = [
                torchmetrics.Accuracy(task=task, num_classes=n_classes),
                torchmetrics.Precision(task=task, num_classes=n_classes, average="macro"),
                torchmetrics.Specificity(task=task, num_classes=n_classes, average="macro"),
                torchmetrics.Recall(task=task, num_classes=n_classes, average="macro"),
                torchmetrics.F1Score(task=task, num_classes=n_classes, average="macro"),
                torchmetrics.PrecisionRecallCurve(task=task, num_classes=n_classes),
                torchmetrics.AUROC(task=task, num_classes=n_classes),
                torchmetrics.ROC(task=task, num_classes=n_classes),
                torchmetrics.ConfusionMatrix(task=task, num_classes=n_classes),
            ]

        return metrics

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = x.float()
        x = self.pre_conv(x)
        x = self.resnet(x)
        x = self.model(x)
        x = x.squeeze()

        return x

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y, weight = batch

        x = x.float()
        y = y.float()
        weight = weight.float()

        y_hat = self(x)
        loss = self.loss(y_hat, y, weight=weight, reduction="mean")

        for metric in self.metrics:
            self.log("train__"+metric.__class__.__name__, metric(y_hat, y), on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int):

        x, y, weight = batch

        x = x.float()
        y = y.float()
        weight = weight.float()

        y_hat = self(x)
        loss = self.loss(y_hat, y, weight=weight, reduction="mean")
        
        for metric in self.metrics:
            self.log("train__"+metric.__class__.__name__, metric(y_hat, y), on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def loss(self, y_hat: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, **kwargs) -> torch.Tensor:

        if self.n_classes == 1:
            return F.binary_cross_entropy(y_hat, y, weight=weight, **kwargs)
        else:
            return F.cross_entropy(y_hat, y, weight=weight, **kwargs)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)