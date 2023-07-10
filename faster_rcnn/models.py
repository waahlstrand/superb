
from typing import Any, Tuple
import pytorch_lightning as L
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn

from typing import *
from faster_rcnn.augmentation import DetectionAugmentation
import matplotlib.pyplot as plt
import numpy as np
from torchvision.ops import box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics import Accuracy, Precision, Recall

def plot_image(x, targets, predictions):

    x = x.cpu()

    f, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(x.permute(1, 2, 0))
    ax[1].imshow(x.permute(1, 2, 0))

    # Plot boxes and keypoints
    for bbox, label in zip(targets["boxes"], targets["labels"]):

        bbox = bbox.cpu()
        ax[0].add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False, color="g"))
        ax[0].text(bbox[0] + 50, bbox[1], f"{label}", color="g")


    for bbox_hat, label_hat in zip(predictions["boxes"], predictions["labels"]):

        # Convert to cpu
        bbox_hat = bbox_hat.cpu()
        ax[1].add_patch(plt.Rectangle((bbox_hat[0], bbox_hat[1]), bbox_hat[2] - bbox_hat[0], bbox_hat[3] - bbox_hat[1], fill=False, color="red"))
        ax[1].text(bbox_hat[0] + 50, bbox_hat[1], f"{label_hat}", color="red")

    if "keypoints" in targets.keys() and "keypoints" in predictions.keys():
        for keypoint in targets["keypoints"]:
                
            # Convert to cpu
            keypoint = keypoint.cpu()
            ax[0].scatter(keypoint[:,0], keypoint[:,1], color="g")
        
        for keypoint_hat in predictions["keypoints"]:
            
            keypoint_hat = keypoint_hat.cpu()
            ax[1].scatter(keypoint_hat[:,0], keypoint_hat[:,1], color="red")

    return f, ax


class SuperbDetector(L.LightningModule):

    def __init__(self, 
                 model: nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 augmentation: DetectionAugmentation, 
                 optimizer_params: Dict[str, Any] = {},
                 training_params: Dict[str, Any] = {},
                 labels: bool = True,
                 *args: Any, **kwargs: Any,) -> None:
        
        super().__init__(*args, **kwargs)

        self.model = model
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.training_params = training_params
        self.augmentation = augmentation

        self.map = MeanAveragePrecision(num_classes=4, class_metrics=True)

        # self.label_metrics = {
        #     "accuracy": Accuracy(num_classes=4, task="multiclass"),
        #     "precision": Precision(num_classes=4, task="multiclass"),
        #     "recall": Recall(num_classes=4, task="multiclass")
        # }

        self.labels = labels

        self.save_hyperparameters(
            "optimizer_params",
            "training_params",
        )

        self.validation_step_outputs = []
        self.training_step_outputs = []

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx) -> STEP_OUTPUT:

        x, y = batch

        if not self.labels:
            y = [{"labels": torch.zeros_like(_["labels"]), "boxes": _["boxes"], "keypoints": _["keypoints"]} for _ in y]

            # print(y["labels"])
        
        outs = self._step(x, y, augment=self.training)

        if batch_idx == 0:

            # Make figure
            f, ax = plot_image(outs["x"][0], outs["y"][0], outs["y"][0])
            self.logger.experiment.add_figure("training examples", f, self.current_epoch)

        self.log("train_loss", outs["loss"].detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.training_params["batch_size"])

        return outs
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx) -> STEP_OUTPUT:

        x, targets = batch
        x = self.augmentation.normalize(x)

        predictions = self.model(x)

        self.map.update(predictions, targets)

        mean_average_precision = self.map.compute()

        self.map.reset()

        # Plot image
        if batch_idx == 0:
            # Print to log

            # Make figure
            f, ax = plot_image(x[0], targets[0], predictions[0])
            self.logger.experiment.add_figure("predictions", f, self.current_epoch)

        self.log("val_map", mean_average_precision["map"], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.training_params["batch_size"])

        for i, map_per_class in enumerate(mean_average_precision["map_per_class"]):
            self.log(f"val_map_{i}", map_per_class, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=self.training_params["batch_size"])

        self.validation_step_outputs.append(
            {
                "y": targets,
                "y_hat": predictions
            }
        )


    
    def on_validation_epoch_end(self) -> None:
    
        # Flatten outputs
        targets = [item for sublist in self.validation_step_outputs for item in sublist["y"]]
        predictions = [item for sublist in self.validation_step_outputs for item in sublist["y_hat"]]

        # Compute mean average precision
        self.map.update(predictions, targets)

        mean_average_precision = self.map.compute()

        self.map.reset()

        self.log("val_map", mean_average_precision["map"], on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.training_params["batch_size"])

        # Reset outputs
        self.validation_step_outputs = []

    
    def configure_optimizers(self) -> torch.optim.Optimizer:

        optimizer = self.optimizer(self.parameters(), **self.optimizer_params)

        return optimizer

    def _step(self, x: torch.Tensor, y: List[Dict[str, torch.Tensor]], augment: bool = True) -> torch.Tensor:
        
        # Augment data
        if augment:
            x, y = self.augmentation(x, y)

        # Process targets
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in y]

        loss_dict = self.model(x, targets)

        losses = sum(loss for loss in loss_dict.values())

        return {"loss": losses, "log": loss_dict, "x": x, "y": y}
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model(x)
    
def compute_iou(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute intersection over union between two boxes

    Args:
        y_hat (torch.Tensor): [description]
        y (torch.Tensor): [description]

    Returns:
        torch.Tensor: [description]
    """

    if len(y_hat) == 0 and len(y) > 0:
        iou = torch.tensor(0, device=y.device, dtype=y_hat.dtype)

    elif len(y_hat) > 0 and len(y) == 0:
        iou = torch.tensor(0, device=y.device, dtype=y_hat.dtype)

    elif len(y_hat) == 0 and len(y) == 0:
        iou = torch.tensor(1, device=y.device, dtype=y_hat.dtype)
    else:
        iou = box_iou(y_hat, y).diag().mean()

    return iou