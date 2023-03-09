#%%
import pytorch_lightning as pl
from preprocessing.images import PADDING_SHAPE
from superb import SuperbDataset
from pathlib import Path
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import warnings
from pytorch_lightning import loggers as pl_loggers
from typing import *
from models import SuperbModel
warnings.filterwarnings("ignore")

#%%
# Constants
DATA_ROOT = Path("")
LABEL_TYPE = "categorical"
N_CHANNELS = 1
N_CLASSES = 14
N_WORKERS = 8

# Hyperparameters
lr = 1e-6
n_epochs = 100
train_batch_size = 256
val_batch_size = 256

# Logging
tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")

# Dataloading
dataset = SuperbDataset(DATA_ROOT, LABEL_TYPE)
train_size  = int(0.85 * len(dataset))
val_size    = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader    = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=N_WORKERS, shuffle=True)
val_dataloader      = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=N_WORKERS, shuffle=True)

#%%
# Training
model = SuperbModel(N_CHANNELS, N_CLASSES, lr=lr)
trainer = pl.Trainer(gpus=1, max_epochs=n_epochs, logger=tb_logger)
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
