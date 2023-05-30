
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
from models.augmentations import Augmentation, DetectionAugmentation
import torchmetrics
import matplotlib.pyplot as plt
import numpy as np
from torchvision.ops import box_iou, generalized_box_iou

class SuperbDetector(L.LightningModule):

    def __init__(self, 
                 model: nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 augmentation: DetectionAugmentation, 
                 optimizer_params: Dict[str, Any] = {},
                 training_params: Dict[str, Any] = {},
                 *args: Any, **kwargs: Any,) -> None:
        
        super().__init__(*args, **kwargs)

        self.model = model
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.training_params = training_params
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
        
        outs = self._step(x, y, augment=self.training)

        self.training_step_outputs.append(outs["loss"])

        self.log("train_loss", outs["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return outs
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx) -> STEP_OUTPUT:

        x, targets = batch
        predictions = self.model(x)

        # print(x.shape)
        # print("predictions", predictions)
        # print("targets", targets)
        # print(predictions)

        y_hat_boxes = [y_hat["boxes"] for y_hat in predictions]
        y_boxes = [y["boxes"] for y in targets]
        # ys = [y["labels"] for y in targets]

        # Check all boxes are valid
        ious = []
        for y_hat, y in zip(y_hat_boxes, y_boxes):
            assert torch.all(y_hat[:, 0] < y_hat[:, 2])
            assert torch.all(y_hat[:, 1] < y_hat[:, 3])
            assert torch.all(y[:, 0] < y[:, 2])
            assert torch.all(y[:, 1] < y[:, 3])

            if len(y_hat) == 0 and len(y) > 0:
                iou = torch.tensor(0, device=y.device, dtype=y_hat.dtype)

            elif len(y_hat) > 0 and len(y) == 0:
                iou = torch.tensor(0, device=y.device, dtype=y_hat.dtype)

            elif len(y_hat) == 0 and len(y) == 0:
                iou = torch.tensor(1, device=y.device, dtype=y_hat.dtype)
            else:
                iou = box_iou(y_hat, y).diag().mean()

            ious.append(iou)
        
        iou = torch.stack(ious).mean()


        self.log("val_iou", iou, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.validation_step_outputs.append(iou)

        return {"val_iou": iou}
    
    def on_validation_epoch_end(self) -> None:
    
        # Compute mean of all validation step outputs
        mean_iou = torch.stack(self.validation_step_outputs).mean()

        # Reset outputs
        self.validation_step_outputs = []

        return {"avg_val_iou": mean_iou}
    
    # def test_step(self, batch) -> STEP_OUTPUT:

    #     loss = self._step(*batch, augment=self.training)

    #     return loss
    
    
    def configure_optimizers(self) -> torch.optim.Optimizer:

        optimizer = self.optimizer(self.parameters(), **self.optimizer_params)

        return optimizer

    def _step(self, x: torch.Tensor, y: List[Dict[str, torch.Tensor]], augment: bool = True) -> torch.Tensor:
        
        # Augment data
        if augment:
            x = self.augmentation(x)

        # Process targets
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in y]

        loss_dict = self.model(x, targets)

        losses = sum(loss for loss in loss_dict.values())

        return {"loss": losses, "log": loss_dict, "x": x}
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model(x)
    
