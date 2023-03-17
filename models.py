import torch
from torch.nn import functional as F
import torch.nn as nn
from torchvision.models import resnet18, resnet34, efficientnet_v2_l
import pytorch_lightning as pl
from preprocessing.images import PADDING_SHAPE
import warnings
import torchmetrics
from typing import *
from sklearn.utils import class_weight
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

        self.save_hyperparameters()

        super().__init__()

        self.metrics = nn.ModuleList(self.set_metrics("binary", n_classes) if n_classes == 1 else self.set_metrics("multiclass", n_classes))
        self.loss = nn.BCEWithLogitsLoss()
        self.pre_conv = nn.Conv2d(in_channels=n_channels, out_channels=3, kernel_size=self.kernel, stride=1, padding=0)
        self.resnet = resnet34(pretrained=False)
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1000, n_classes),
            nn.Sigmoid()
        )

    def set_metrics(self, task, n_classes: int) -> List[torchmetrics.Metric]:

        metrics = [
                torchmetrics.Accuracy(task=task, num_classes=n_classes),
                torchmetrics.Precision(task=task, num_classes=n_classes, average="macro"),
                torchmetrics.Specificity(task=task, num_classes=n_classes, average="macro"),
                # torchmetrics.Recall(task=task, num_classes=n_classes, average="macro"),
                # torchmetrics.F1Score(task=task, num_classes=n_classes, average="macro"),
                # torchmetrics.PrecisionRecallCurve(task=task, num_classes=n_classes),
                # torchmetrics.AUROC(task=task, num_classes=n_classes),
                # torchmetrics.ROC(task=task, num_classes=n_classes),
                # torchmetrics.ConfusionMatrix(task=task, num_classes=n_classes),
            ]
        
        metrics = [metric.to(self.device) for metric in metrics]

        return metrics

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = x.float()
        x = self.pre_conv(x)
        x = F.relu(x)
        x = self.resnet(x)
        x = self.model(x)
        x = x.view(-1, self.n_classes)

        return x

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y, weight = batch

        x = x.float()
        y = y.float()
        weight = weight.float()

        y_hat = self(x)

        y = y.squeeze()
        y_hat = y_hat.squeeze()
        loss = self.loss(y_hat, y, reduction="mean")

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

        y = y.squeeze()
        y_hat = y_hat.squeeze()
        loss = self.loss(y_hat, y, reduction="mean")
        
        for metric in self.metrics:
            self.log("train__"+metric.__class__.__name__, metric(y_hat, y), on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    


class SimSiam(pl.LightningModule):

    def __init__(self, dim: int = 2048, prediction_dim: int = 512, lr: float = 0.05, momentum: float = 0.9, weight_decay: float = 1e-6, n_epochs: int = 100, **kwargs):

        super().__init__()
        
        self.dim = dim
        self.init_lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs

        self.save_hyperparameters()

        self.encoder = resnet34(pretrained=False)
        previous_dim = self.encoder.fc.in_features

        # Update the3 encoder's fc layer to output the desired dimension
        self.encoder.fc = nn.Sequential(
            nn.Linear(previous_dim, previous_dim, bias=False),
            nn.BatchNorm1d(previous_dim),
            nn.ReLU(inplace=True),
            nn.Linear(previous_dim, previous_dim, bias=False),
            nn.BatchNorm1d(previous_dim),
            nn.ReLU(inplace=True),
            self.encoder.fc,
            nn.BatchNorm1d(dim, affine=False),
        )
        self.encoder.fc[6].bias.requires_grad_(False)

        self.predictor = nn.Sequential(
            nn.Linear(dim, prediction_dim, bias=False), 
            nn.BatchNorm1d(prediction_dim),
            nn.ReLU(inplace=True), 
            nn.Linear(prediction_dim, dim)
            )
        
        self.criterion = nn.CosineSimilarity(dim=1)

    def configure_optimizers(self) -> Any:
        
        optimizer = torch.optim.SGD(self.parameters(), lr=self.init_lr, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs)

        return optimizer, scheduler
        
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:

        x1, x2 = batch
        x1 = x1.float()
        x2 = x2.float()

        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        loss = -(self.criterion(p1, z2.detach()) + self.criterion(p2, z1.detach())) / 2

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

