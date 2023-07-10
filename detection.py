import json
import argparse
import warnings
import time
from tqdm import tqdm
import matplotlib
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Subset, DataLoader

from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from torchvision import models


from models.augmentations import Augmentation
from models.augmentation.detection import DetectionAugmentation
from models.detection import SuperbDetector
from data.superb import Superb, collate_with_bboxes

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
    WEIGHT_DECAY        = 5e-4
    SEED                = 42
    DEVICE              = 'gpu'
    GPUS                = "0"
    N_EPOCHS            = 100
    BACKBONE            = 'faster-rcnn'
    LOG_DIR             = './logs'
    LABEL_TYPE          = 'binary'
    NAME                = 'superb'
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
    parser.add_argument('--gpus', type=str, default=GPUS)
    parser.add_argument('--n_epochs', type=int, default=N_EPOCHS)
    parser.add_argument('--backbone', type=str, default=BACKBONE)
    parser.add_argument('--log_dir', type=str, default=LOG_DIR)
    parser.add_argument('--name', type=str, default=NAME)
    parser.add_argument('--severity', type=int, default=SEVERITY)
    parser.add_argument('-s', '--resize_shape', nargs='+', type=int, default=None)
    parser.add_argument('-d', '--debug', type=bool, default=False)
    parser.add_argument('--no_fractures', type=bool, default=True)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--target', type=str, default='keypoint')
    parser.add_argument('--labels', type=int, default=1)

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
    
        # csv_logger = CSVLogger(args.log_dir, name=args.name)
        tb_logger  = TensorBoardLogger(args.log_dir, name=args.name)

        loggers = [tb_logger]

        model_dir  = tb_logger.log_dir + "/checkpoints"
        # checkpoint = ModelCheckpoint(model_dir, monitor='val_iou', save_top_k=3, mode='max')
        checkpoint = ModelCheckpoint(model_dir, monitor='train_loss', save_top_k=3, mode='min')

        callbacks = [checkpoint]


    # Load data
    # Read config
    with open(args.cfg, 'r') as f:
        config = json.load(f)

    errors = pd.read_csv("reconstruction_errors.csv")
    error_moid   = errors.moid.values


    data_root       = Path(args.source)
    removed_samples = config["removed"] + error_moid.tolist()
    shape           = config["large_shape"] #if not args.resize_shape else args.resize_shape

    ds = Superb(
        Path(data_root),
        size=shape,
        removed=removed_samples,
        severity=args.severity,
        mode="severity",
        dtype=np.float32
    )

    has_full_spine = lambda p: all([v.coordinates is not None for v in p.vertebrae])

    ds = ds.filter(has_full_spine)

    # Undersample dataset
    # print(f"Number of positive samples: {len(has_annotation)}")
    # print(f"Number of negative samples: {len(has_no_fracture)}")

    # Randomly sample from the majority class
    # undersampled_idx = np.random.choice(has_no_fracture, len(has_annotation), replace=False)

    # if args.no_fractures:
        # idxs = np.concatenate((has_annotation, undersampled_idx))
    # else:
        # idxs = has_annotation

    idxs = np.arange(len(ds))

    # Split into train and validation
    train_idx, val_idx = train_test_split(idxs, test_size=1-args.train_fraction, random_state=args.seed, shuffle=True)

    
    train_dataset       = Subset(ds, train_idx)
    validation_dataset  = Subset(ds, val_idx)

    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")

    train_dataloader    = DataLoader(train_dataset, 
                                     batch_size=args.batch_size, 
                                     num_workers=args.n_workers, 
                                     shuffle=True, 
                                     collate_fn=lambda x: collate_with_bboxes(ds, x))
    val_dataloader      = DataLoader(validation_dataset, 
                                     batch_size=args.batch_size, 
                                     num_workers=args.n_workers, 
                                     shuffle=False, 
                                     collate_fn=lambda x: collate_with_bboxes(ds, x))

    ############################################################################################################
    if args.labels:
        n_classes = 4
    else:
        n_classes = 1

    if args.target == "keypoint":
        backbone    = models.detection.keypointrcnn_resnet50_fpn(
            pretrained=False, 
            progress=True, 
            num_classes=n_classes+1, 
            num_keypoints=6,
            pretrained_backbone=True,
            #trainable_backbone_layers=5

            )
    elif args.target == "bbox":

        backbone    = models.detection.fasterrcnn_resnet50_fpn(
            pretrained=False, 
            progress=True, 
            num_classes=n_classes, 
            box_detections_per_img=20,
            pretrained_backbone=True)
    else:
        raise ValueError(f"Target {args.target} not supported")
    
    if args.optimizer == 'adam':
        optimizer   = torch.optim.Adam
        optimizer_params = {
            'lr': args.lr,
            'weight_decay': args.weight_decay
        }
    elif args.optimizer == 'sgd':
        optimizer   = torch.optim.SGD
        optimizer_params = {
            'lr': args.lr,
            'momentum': args.momentum,
            'weight_decay': args.weight_decay
        }
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    p_augmentation  = 0.5
    p_dropout       = 0.2
    p_resize        = 0.0
    augmentation    = DetectionAugmentation(
        p=p_augmentation, 
        size_percent=(0.04, 0.04), 
        p_dropout=p_dropout,
        p_resize=p_resize
        )

    model = SuperbDetector(
        backbone, 
        optimizer, 
        augmentation, 
        optimizer_params=optimizer_params,
        training_params={
            'backbone': backbone.__class__.__name__,
            'n_epochs': args.n_epochs, 
            'batch_size': args.batch_size,
            'n_classes': n_classes,
            },
            labels=args.labels,
        )
    
    trainer = pl.Trainer(
        accelerator=args.device, 
        devices=[int(gpu) for gpu in args.gpus.split(",")],
        max_epochs=args.n_epochs, 
        logger=loggers, 
        callbacks=callbacks, 
        log_every_n_steps=10)

    # Train model
    torch.set_float32_matmul_precision('medium')
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    if not args.debug:
        trainer.save_checkpoint(f"{model_dir}/{name}_final.ckpt")


if __name__ == "__main__":


    main()