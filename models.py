import torch
from torch.nn import functional as F
from torchvision import transforms as T
import torch.nn as nn
from torchvision.models import resnet18, resnet34, efficientnet_b7
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import matplotlib.pyplot as plt
from preprocessing.images import PADDING_SHAPE
import warnings
import torchmetrics
from typing import *
from augmentations import Augmenter, Preprocess
warnings.filterwarnings("ignore")

def plot_prediction_histogram(preds, labels, title):
    fig, ax = plt.subplots(figsize=(10, 10))

    preds_true = preds[labels == 1]
    preds_false = preds[labels == 0]

    ax.hist(preds_true, bins=50, label="predicted (true)", density=True, color="green", alpha=0.3)
    ax.hist(preds_false, bins=50, label="predicted (false)", density=True, color="red", alpha=0.3)

    ax.hist(labels, bins=2, label="labels", density=True, color="gray", alpha=0.8)
    ax.set_title(title)
    ax.legend()
    
    return fig

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, activation = nn.ELU()):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = 1, padding = 0)
        self.activation = activation
        self.pooling = nn.AvgPool2d((2, 2))

    def forward(self, x):

        x = self.conv1(x)
        x = self.activation(x)
        x = self.pooling(x)

        return x

class CustomSuperbBackbone(nn.Module):

    def __init__(self, n_classes: int = 1):

        self.n_classes = n_classes

        super().__init__()

        self.model = nn.Sequential(
            ConvBlock(3, 16, 3),
            *[ConvBlock(16*(2**i), 16*(2**(i+1)), 3) for i in range(7)],
            nn.Flatten(),
            nn.Linear(8192, 2048),
            nn.ELU(),
            nn.Linear(2048, 1024),
            nn.ELU(),
            nn.Linear(1024, self.n_classes),
        )

    def forward(self, x):
            
            return self.model(x)


class SuperbModel(pl.LightningModule):

    def __init__(self, backbone: nn.Module,
                       augmenter: Augmenter,
                       optimizer: torch.optim.Optimizer,
                       n_classes: int
                       ):



        super().__init__()

        self.n_classes = n_classes
        self.optimizer = optimizer

        # self.save_hyperparameters()

        self.backbone = backbone
        self.task           = "binary" if n_classes == 1 else "multilabel"
        self.augmenter      = augmenter
        self.pre_conv       = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(2, 2), stride=1, padding=0), nn.ELU())
        self.model          = backbone
        self.loss           = F.binary_cross_entropy_with_logits if self.task == "binary" else F.multilabel_soft_margin_loss
        self.metrics        = nn.ModuleList(self.get_metrics(self.task, n_classes))
        self.preprocess     = Preprocess()
        self.val_augmenter  = Augmenter(p=0)

    def get_metrics(self, task, n_classes: int) -> List[torchmetrics.Metric]:

        metrics = [
                torchmetrics.Accuracy(task=task, num_classes=n_classes),
                torchmetrics.Precision(task=task, num_classes=n_classes, average="macro"),
                torchmetrics.Specificity(task=task, num_classes=n_classes, average="macro"),
                torchmetrics.Recall(task=task, num_classes=n_classes, average="macro"),
                torchmetrics.F1Score(task=task, num_classes=n_classes, average="macro"),
                # torchmetrics.PrecisionRecallCurve(task=task, num_classes=n_classes),
                torchmetrics.AUROC(task=task, num_classes=n_classes),
                # torchmetrics.ROC(task=task, num_classes=n_classes),
                # torchmetrics.ConfusionMatrix(task=task, num_classes=n_classes),
            ]
        
        metrics = [metric.to(self.device) for metric in metrics]

        return metrics

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.pre_conv(x)
        # x = x.repeat(1, 3, 1, 1)
        x = self.model(x)
        x = x.view(-1, self.n_classes)

        return x

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:

        x, y, w = batch
        x, y, w = x.unsqueeze(1), y.unsqueeze(1), w.unsqueeze(1)
        w = torch.ones_like(y, dtype=torch.float)
        
        x = self.augmenter(x)
        y_hat = self(x)

        loss = self.loss(y_hat, y.float(), weight=w)

        self.log("train_loss", loss.cpu().item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx % 100 == 0:
            self.logger.experiment.add_image("train_image", x[0, 0, :, :], self.current_epoch, dataformats="HW")

        y_preds = torch.sigmoid(y_hat.detach())

        return {"loss": loss, "y_hat": y_preds, "y": y}
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):

        x, y, w = batch
        x, y, w = x.unsqueeze(1), y.unsqueeze(1), w.unsqueeze(1)
        w = torch.ones_like(y, dtype=torch.float)

        x = self.val_augmenter(x)
    
        y_hat = self(x)

        loss = self.loss(y_hat, y.float(), weight=w)
        y_preds = torch.sigmoid(y_hat)
        
        return {"loss": loss, "y_hat": y_preds, "y": y}
    
    def training_epoch_end(self, outputs: Dict[str, torch.Tensor]):

        y_true = torch.cat([o['y'] for o in outputs])
        y_pred = torch.cat([o['y_hat'] for o in outputs])
        
        for metric in self.metrics:
            self.log("train_"+metric.__class__.__name__, metric(y_pred.cpu(), y_true.cpu().float()), prog_bar=False, logger=True)


    def validation_epoch_end(self, outputs: Dict[str, torch.Tensor]):


        loss = torch.stack([o['loss'] for o in outputs]).mean()
        y_true = torch.cat([o['y'] for o in outputs])
        y_pred = torch.cat([o['y_hat'] for o in outputs])

        self.log("val_loss", loss.cpu().item(), on_epoch=True, prog_bar=True, logger=True)
        

        for metric in self.metrics:
            self.log("val_"+metric.__class__.__name__, metric(y_pred.cpu(), y_true.cpu().float()), prog_bar=False, logger=True)

        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_figure("predictions", plot_prediction_histogram(y_pred.cpu().flatten(), y_true.cpu().flatten(), title="Test"), self.current_epoch)


    def configure_optimizers(self) -> torch.optim.Optimizer:
        # return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        return self.optimizer
    

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

