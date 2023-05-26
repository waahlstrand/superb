
from typing import Any, Tuple
import pytorch_lightning as L
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torchvision import transforms as T
from typing import *
from models.augmentations import Augmentation, SimSiamAugmentation, SameRandomCrop
import torchmetrics
import matplotlib.pyplot as plt
import numpy as np

class SuperbModel(L.LightningModule):

    def __init__(self, 
                 model: nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 loss: nn.Module, 
                 augmentation: Augmentation, 
                 optimizer_params: Dict[str, Any] = {},
                 training_params: Dict[str, Any] = {},
                 *args: Any, **kwargs: Any,) -> None:
        
        super().__init__(*args, **kwargs)

        self.model = model
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.training_params = training_params
        self.loss = loss
        self.augmentation = augmentation
        self.metrics        = nn.ModuleList([
                torchmetrics.Accuracy(task="binary"),
        ])

        self.save_hyperparameters(
            "optimizer_params",
            "training_params",
        )

        self.validation_step_outputs = []
        self.training_step_outputs = []

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx) -> STEP_OUTPUT:

        x, y = batch
        
        loss, y_hat, x_hat = self._step(x, y, augment=self.training)

        preds = torch.sigmoid(y_hat)

        # Save outputs for later
        self.training_step_outputs.append({
            "loss": loss.detach().cpu(),
            "x": x.detach().cpu(),
            "x_hat": x_hat.detach().cpu(),
            "y": y.detach().cpu(),
            "preds": preds.detach().cpu(),
        })

        return loss

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        
        # Logging
        self._log(self.training_step_outputs, "train")

        # Plot sample images
        if batch_idx % 100 == 0:
            n_samples = 4
            fig, axs = plt.subplots(1, n_samples, figsize=(n_samples*2, 4))
            for i in range(n_samples):
                axs[i].imshow(self.training_step_outputs[-1]["x_hat"][i].permute(1,2,0).numpy())
                axs[i].set_title(f"p: {self.training_step_outputs[-1]['preds'][i].item():.2f}, y: {self.training_step_outputs[-1]['y'][i].item():.2f}")
                
                axs[i].axis("off")

            self.logger.experiment.add_figure("train/sample", fig, global_step=self.global_step)
            
        # Reset outputs
        self.training_step_outputs = []
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx) -> STEP_OUTPUT:

        x, y = batch
        loss, y_hat, x_hat = self._step(x, y, augment=self.training)

        preds = torch.sigmoid(y_hat)

        # Save outputs for later
        self.validation_step_outputs.append({
            "loss": loss.detach().cpu(),
            "x": x.detach().cpu(),
            "y": y.detach().cpu(),
            "preds": preds.detach().cpu(),
        })

        return loss
    
    def on_validation_epoch_end(self) -> None:
        
        # Logging
        self._log(self.validation_step_outputs, "val")

        # Reset outputs
        self.validation_step_outputs = []
    
    def test_step(self, batch) -> STEP_OUTPUT:

        loss, preds = self._step(*batch, augment=self.training)

        return loss
    
    
    def configure_optimizers(self) -> torch.optim.Optimizer:

        optimizer = self.optimizer(self.parameters(), **self.optimizer_params)

        return optimizer

    def _step(self, x: torch.Tensor, y: torch.Tensor, augment: bool = True) -> torch.Tensor:
        
        # Augment data
        if augment:
            x = self.augmentation(x)

        # Process targets
        y = torch.minimum(y, torch.ones_like(y)).unsqueeze(1).float()

        y_hat = self.model(x)

        loss  = self.loss(y_hat, y)

        return loss, y_hat, x
    
    def _log(self, outputs: List[Dict[str, float]], namespace: str):

        # Compute outputs
        preds = torch.cat([o["preds"] for o in outputs]).squeeze()
        y = torch.cat([o["y"] for o in outputs]).squeeze()
        y = torch.minimum(y, torch.ones_like(y)).float()
        loss = torch.stack([o["loss"] for o in outputs]).mean()

        # Log metrics
        for metric in self.metrics:
            m = metric(preds, y)
            self.log(f"{namespace}_{metric.__class__.__name__}", m, prog_bar=True, logger=True)

        # Log loss
        self.log(f"{namespace}_loss", loss, prog_bar=True, logger=True)
