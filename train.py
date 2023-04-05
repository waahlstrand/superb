#%%
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torchvision import transforms as T
from torchvision import models
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt
from superb import SuperbDataset
from pathlib import Path
import warnings

from typing import *
from models import SuperbModel
from augmentations import BaseAugmenter, Augmenter
import json
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")

#%%
# Constants
with open("configs/data.json", "r") as f:
        data_config = json.load(f)

DATA_ROOT               = Path(data_config["data_root"])
REMOVED                 = data_config["removed"]
RESIZE_SHAPE            = data_config["min_shape"]
LABEL_TYPE              = "binary"
CLASS_DISTRIBUTION      = data_config["class_distribution"][LABEL_TYPE]
N_CHANNELS              = 1
N_CLASSES               = 1 #13
N_WORKERS               = 8
TRAIN_FRACTION          = 0.85

# Hyperparameters
lr               = 1e-4
momentum         = 0.9
weight_decay     = 1e-2
n_epochs         = 1000
train_batch_size = 8
val_batch_size   = 16
weights          = torch.tensor(list(CLASS_DISTRIBUTION.values()))

# Logging
tb_logger       = pl_loggers.TensorBoardLogger(save_dir="logs/")
csv_logger      = pl_loggers.CSVLogger(save_dir="logs/")
checkpoint      = ModelCheckpoint("./models", monitor='val_loss', save_top_k=2, mode='min')

# Dataloading
dataset         = SuperbDataset(DATA_ROOT, LABEL_TYPE, RESIZE_SHAPE, removed=REMOVED, class_distribution=CLASS_DISTRIBUTION)
train_size      = int(TRAIN_FRACTION * len(dataset))
val_size        = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader    = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=N_WORKERS, shuffle=True, pin_memory=True)
val_dataloader      = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=N_WORKERS, shuffle=False, pin_memory=True)

#%%
# Training
torch.set_float32_matmul_precision('high')
# backbone = models.efficientnet_b7(pretrained=False, num_classes=N_CLASSES)
backbone = models.resnet50(pretrained=False, num_classes=N_CLASSES)

augmenter = Augmenter(p=0.5)
model     = SuperbModel(backbone, augmenter, N_CHANNELS, N_CLASSES, lr=lr, momentum=momentum, weight_decay=weight_decay)
trainer   = pl.Trainer(accelerator="gpu", max_epochs=n_epochs, logger=[tb_logger, csv_logger], callbacks=[checkpoint], log_every_n_steps=5)
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


# %%
