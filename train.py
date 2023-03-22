#%%
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt
from superb import SuperbDataset
from pathlib import Path
import warnings

from typing import *
from models import SuperbModel
from augmentations import Augmenter
import json
warnings.filterwarnings("ignore")

#%%
# Constants
with open("configs/data.json", "r") as f:
        data_config = json.load(f)

DATA_ROOT               = Path(data_config["data_root"])
REMOVED                 = data_config["removed"]
RESIZE_SHAPE            = (600, 633)
LABEL_TYPE              = "binary"
CLASS_DISTRIBUTION      = data_config["class_distribution"][LABEL_TYPE]
N_CHANNELS              = 1
N_CLASSES               = 1 #13
N_WORKERS               = 8
BACKBONE                = "efficientnet-b7"
TRAIN_FRACTION          = 0.85

# Hyperparameters
lr               = 1e-4
n_epochs         = 100
train_batch_size = 3
val_batch_size   = 3
weights          = torch.tensor(list(CLASS_DISTRIBUTION.values()))

# Logging
tb_logger       = pl_loggers.TensorBoardLogger(save_dir="logs/")
checkpoint      = ModelCheckpoint("./models", monitor='val_loss', save_top_k=2, mode='min')

# Dataloading
dataset         = SuperbDataset(DATA_ROOT, LABEL_TYPE, RESIZE_SHAPE, removed=REMOVED)
train_size      = int(TRAIN_FRACTION * len(dataset))
val_size        = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader    = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=N_WORKERS, shuffle=True)
val_dataloader      = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=N_WORKERS, shuffle=True)

#%%
# Training
# torch.set_float32_matmul_precision('medium')

augmenter = Augmenter(p=0.5)
model     = SuperbModel(BACKBONE, augmenter, N_CHANNELS, N_CLASSES, lr=lr, weight=weights)
trainer   = pl.Trainer(gpus=1, max_epochs=n_epochs, logger=tb_logger, callbacks=[checkpoint])
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


# %%
