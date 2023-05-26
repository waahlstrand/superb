from models.contrastive import SimSiam
from models.augmentations import Augmentation
from slask.superb import BinaryDataset
from torch.utils.data import Subset, DataLoader
import pytorch_lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
from pathlib import Path
from sklearn.model_selection import train_test_split
from torchvision import models
import json
import argparse
import warnings
import time
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings("ignore")

def main():
    
    parser = argparse.ArgumentParser(description='Train a PACBED model')

    CONFIG_PATH         = './configs/data.json'
    BATCH_SIZE          = 2
    N_WORKERS           = 8
    TRAIN_FRACTION      = 0.85
    LR                  = 1e-4
    MOMENTUM            = 0.9
    WEIGHT_DECAY        = 1e-2
    SEED                = 42
    DEVICE              = 'gpu'
    N_DEVICES           = 1
    N_EPOCHS            = 100
    BACKBONE            = 'resnet34'
    LOG_DIR             = './logs'
    LABEL_TYPE          = 'binary'
    NAME                = 'contrastive'
    SEVERITY            = 0

    parser.add_argument('--source', type=str, default='/data/balder/datasets/superb/patients')
    parser.add_argument('--cfg', type=str, default=CONFIG_PATH)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--n_workers', type=int, default=N_WORKERS)
    parser.add_argument('--train_fraction', type=int, default=TRAIN_FRACTION)
    parser.add_argument('--label_type', type=str, default=LABEL_TYPE)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--momentum', type=float, default=MOMENTUM)
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--device', type=str, default=DEVICE)
    parser.add_argument('--n_devices', type=str, default=N_DEVICES)
    parser.add_argument('--n_epochs', type=int, default=N_EPOCHS)
    parser.add_argument('--backbone', type=str, default=BACKBONE)
    parser.add_argument('--log_dir', type=str, default=LOG_DIR)
    parser.add_argument('--name', type=str, default=NAME)
    parser.add_argument('--severity', type=int, default=SEVERITY)
    parser.add_argument('-s', '--resize_shape', nargs='+', type=int, default=None)
    parser.add_argument('-d', '--debug', type=bool, default=False)


    args = parser.parse_args()

    # Set seed
    seed_everything(args.seed)   

    # Human readable time
    name = time.strftime("%Y%m%d-%H%M%S")


    # Set up logging
    # Check if logging dir exists
    if not Path(args.log_dir).exists():
        Path(args.log_dir).mkdir(parents=True)

    # If debugging, do not log
    loggers = []
    callbacks = []
    if args.debug:
        print("Debugging, not logging")
    else:
    
        csv_logger = CSVLogger(args.log_dir, name=args.name)
        tb_logger  = TensorBoardLogger(args.log_dir, name=args.name)

        loggers = [csv_logger, tb_logger]

        model_dir  = csv_logger.log_dir + "/checkpoints"
        checkpoint = ModelCheckpoint(model_dir, monitor='val_loss', save_top_k=2, mode='min')

        callbacks = [checkpoint]


    # Load data
    # Read config
    with open(args.cfg, 'r') as f:
        config = json.load(f)

    data_root       = Path(args.source)
    removed_samples = config["removed"]
    shape           = config["min_shape"] if not args.resize_shape else args.resize_shape
    dataset         = BinaryDataset(data_root, shape, removed_samples, severity=args.severity, mode='severity', dtype=np.float32)
    
    # Only select the negative examples
    neg_idx = [i for i, (_, y) in enumerate(dataset) if y == 0]
    print(f"Number of negative samples: {len(neg_idx)}")

    subset = Subset(dataset, neg_idx)

    # Split into train and validation
    train_idx, val_idx  = train_test_split(np.arange(len(subset)), train_size=args.train_fraction, shuffle=True, random_state=args.seed)
    
    train_dataset       = Subset(subset, train_idx)
    validation_dataset  = Subset(subset, val_idx)

    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    
    train_dataloader    = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True)
    val_dataloader      = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)

    # Set up model
    model = SimSiam()

    # Set up trainer
    trainer = L.Trainer(
        accelerator=args.device,
        max_epochs=args.n_epochs,
        logger=loggers,
        callbacks=callbacks)
    
    # Train
    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == "__main__":

    main()