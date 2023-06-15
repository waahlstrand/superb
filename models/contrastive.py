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
from models.augmentations import Augmentation, SimSiamAugmentation, SameRandomCrop, Patchify
import torchmetrics
import matplotlib.pyplot as plt
import numpy as np

class SimSiam(L.LightningModule):

    def __init__(self, 
                 dim: int = 1000, 
                 prediction_dim: int = 512, 
                 lr: float = 0.05, 
                 momentum: float = 0.9, 
                 weight_decay: float = 1e-6, 
                 n_epochs: int = 100, **kwargs):

        super().__init__()
        
        self.dim = dim
        self.init_lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.augmentation = SimSiamAugmentation()

        self.save_hyperparameters()

        self.encoder = models.resnet50(pretrained=False)
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

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
        
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:

        x, y = batch
        x1 = x
        x2 = self.augmentation(x)

        x1 = x1.float()
        x2 = x2.float()

        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        std_z1 = torch.std(z1, dim=0).mean()

        loss = -(self.criterion(p1, z2.detach()) + self.criterion(p2, z1.detach())) / 2
        loss = loss.mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("std_z1", std_z1, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss


class PatchwiseSimSiam(SimSiam):

    def __init__(self, size: Tuple[int, int] = (256, 256), p: float = 0.9, expansion_factor: float=0.3, dim: int = 1000, prediction_dim: int = 512, lr: float = 0.05, momentum: float = 0.9, weight_decay: float = 1e-6, n_epochs: int = 100, **kwargs):

        super().__init__(dim=dim, prediction_dim=prediction_dim, lr=lr, momentum=momentum, weight_decay=weight_decay, n_epochs=n_epochs, **kwargs)
        self.size = size
        self.p = p
        self.expansion_factor = expansion_factor
        self.same_crop = SameRandomCrop(p=p, size=size, expansion_factor=expansion_factor)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:

        # Ignore the label
        x, y = batch

        # Assume batch is twice the actual batch size
        x1, x2 = torch.split(x, x.shape[0] // 2, dim=0)
    
        # Take same crop from both images
        x1, x2 = self.same_crop(x1, x2)
        
        # Augment one of the images
        # x2 = self.augmentation(x2)

        x1 = x1.float()
        x2 = x2.float()

        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        std_p1 = torch.std(torch.nn.functional.normalize(p1, dim=1), dim=0).mean()

        loss = -(self.criterion(p1, z2.detach()) + self.criterion(p2, z1.detach())) / 2
        
        if batch_idx % 100 == 0:
            
            # Log the images in a grid comparing each image in the batch
            idxs = np.random.choice(x1.shape[0], np.maximum(x1.shape[0], 10), replace=False)
            f, ax = plt.subplots(2, np.maximum(x1.shape[0], 10), figsize=(15, 5))
            for i in idxs:
                ax[0, i].imshow(x1[i].detach().cpu().permute(1, 2, 0))
                ax[1, i].imshow(x2[i].detach().cpu().permute(1, 2, 0))

                # Remove axis
                ax[0, i].axis("off")
                ax[1, i].axis("off")

                # Add loss as title on first row
                ax[0, i].set_title(f"{loss[i].item():.2f}")

            # Tight layout
            plt.tight_layout()

            self.logger.experiment.add_figure("train_images", f, global_step=self.global_step)
        
        loss = loss.mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("std_p", std_p1, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

class GridPatchSimSiam(SimSiam):

    def __init__(self, 
                n_patches: int = 16,
                image_size: Tuple[int, int] = (256, 256),
                 dim: int = 1000, 
                 prediction_dim: int = 512, 
                 lr: float = 0.05, 
                 momentum: float = 0.9, 
                 weight_decay: float = 0.000001, 
                 n_epochs: int = 100, **kwargs):
        
        super().__init__(dim, prediction_dim, lr, momentum, weight_decay, n_epochs, **kwargs)

        self.n_patches = n_patches
        self.image_size = image_size
        self.patchify = Patchify.from_n_patches(self.n_patches, self.image_size)
        self.patch_embeddings = PatchEncoding(n_patches=self.n_patches, dim=dim)

    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:

        # Ignore the label
        x, y = batch

        # Assume batch is twice the actual batch size
        x1, x2 = torch.split(x, x.shape[0] // 2, dim=0)

        # Make rectangular patches from images
        x1, pos1 = self.patchify(x1)
        x2, pos2 = self.patchify(x2)

        pos1 = self.patch_embeddings(pos1.to(x1.device))
        pos2 = self.patch_embeddings(pos2.to(x2.device))
        
        # Augment one of the images
        # x2 = self.augmentation(x2)

        x1 = x1.float()
        x2 = x2.float()

        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        # Add positional encoding
        z1 = z1 + pos1
        z2 = z2 + pos2
        
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)


        std_p1 = torch.std(torch.nn.functional.normalize(p1, dim=1), dim=0).mean()

        loss = -(self.criterion(p1, z2.detach()) + self.criterion(p2, z1.detach())) / 2

        loss = loss.mean()

        return loss


class PatchEncoding(nn.Module):

    def __init__(self, n_patches: int = 16, dim: int = 1024):

        super().__init__()

        self.n_patches = n_patches
        self.dim = dim

        self.embedding = nn.Embedding(n_patches, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.embedding(x)

