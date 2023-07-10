import json
import argparse
import warnings
from pathlib import Path

import pandas as pd

import torch
import torchvision
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

from data.superb import SuperbDataModule

from faster_rcnn.models import SuperbDetector
from faster_rcnn.augmentation import DetectionAugmentation
# from detr.models import SuberbDetr

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="Train a SUPERB model")
    parser.add_argument("--source", type=str, default="")
    parser.add_argument("--cfg", type=str, default="")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=0)
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--name", type=str, default="superb")
    parser.add_argument("-d", "--debug", type=bool, default=False)
    parser.add_argument("--target", type=str, default="keypoint")

    # Training parameters
    parser.add_argument("--backbone", type=str, default="faster-rcnn")
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_epochs", type=int, default=500)
    parser.add_argument("--n_workers", type=int, default=24)
    parser.add_argument("--train_fraction", type=int, default=0.85)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--width", type=int, default=280)
    parser.add_argument("--bbox_expansion", type=float, default=0.1)
    parser.add_argument("--bbox_format", type=str, default="xyxy")
    parser.add_argument("--n_classes", type=int, default=4)
    parser.add_argument("--n_keypoints", type=int, default=6)
    parser.add_argument("--p_dropout", type=float, default=0.2)
    parser.add_argument("--p_augmentation", type=float, default=0.5)
    parser.add_argument("--p_crop", type=float, default=0.2)

    parser.add_argument("--checkpoint", type=str, default="")

    # Get arguments
    args = parser.parse_args()

    training_params = {
                'n_epochs': args.n_epochs,
                'batch_size': args.batch_size,
                'n_workers': args.n_workers,
                'train_fraction': args.train_fraction,
                'height': args.height,
                'width': args.width,
                'p_dropout': args.p_dropout,
                'p_augmentation': args.p_augmentation,
                'p_crop': args.p_crop
            }

    # Set seed
    seed_everything(args.seed)

    if not Path(args.log_dir).exists():
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # Handle logging
    if not args.debug:
        tb_logger = TensorBoardLogger(args.log_dir, name=args.name)
        csv_logger = CSVLogger(args.log_dir, name=args.name, version=tb_logger.version)

        loggers = [tb_logger, csv_logger]

        # Handle checkpointing
        checkpoint_callbacks = [ModelCheckpoint(
            Path(tb_logger.log_dir) / "checkpoints",
            monitor="val_map",
            mode="max",
            save_top_k=1,
            # save_last=True,
        )]

    else:
        loggers = []
        checkpoint_callbacks = []

    # Read configuration and get removed samples
    with open(args.cfg, 'r') as f:
        config = json.load(f)

    errors = pd.read_csv(config["errors"])
    error_moid   = errors.moid.values

    data_root       = Path(args.source)
    removed_samples = config["removed"] + error_moid.tolist()

    # Initialize data module
    dm = SuperbDataModule(
        data_dir = data_root,
        batch_size=args.batch_size,
        image_size=(args.height, args.width),
        train_split=args.train_fraction,
        removed=removed_samples,
        n_workers=args.n_workers,
        n_classes=args.n_classes,
        n_keypoints=args.n_keypoints,
        bbox_expansion=args.bbox_expansion,
        bbox_format=args.bbox_format,
        target_format=["label", "bbox", "keypoint"],
        filter=lambda p: all([v.coordinates is not None for v in p.vertebrae])
    )

    augmentation    = DetectionAugmentation(
        p=args.p_augmentation, 
        size_percent=(0.04, 0.04), 
        p_dropout=args.p_dropout,
        p_crop=args.p_crop,
        )

    # Set up loss and optimizer
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

    # Initialize model
    if args.backbone == "faster-rcnn":
        backbone = torchvision.models.detection.keypointrcnn_resnet50_fpn(
            num_classes=dm.n_classes+1,
            num_keypoints=dm.n_keypoints,
            weights_backbone=None,
        )

        model = SuperbDetector(
            model=backbone,
            optimizer=optimizer,
            augmentation=augmentation,
            training_params=training_params,
            optimizer_params=optimizer_params,
        )

    elif args.backbone == "detr":
        backbone = torchvision.models.detr_resnet50(
            num_classes=dm.n_classes+1,
            num_keypoints=dm.n_keypoints,
            pretrained=True
        )

        # model = SuberbDetr(
        #     backbone=backbone,
        #     optimizer=optimizer,
        #     optimizer_params=optimizer_params,
        #     lr=args.lr,
        #     n_classes=dm.n_classes,
        #     n_keypoints=dm.n_keypoints,
        #     target=args.target
        # )

    else:
        raise ValueError(f"Backbone {args.backbone} not supported")
    
    # Initialize trainer
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = [int(gpu) for gpu in args.device.split(",")]
    else:
        accelerator = "cpu"
        devices = None

    if args.checkpoint:
        trainer = Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=args.n_epochs,
            logger=loggers,
            callbacks=checkpoint_callbacks,
            resume_from_checkpoint=args.checkpoint,
            log_every_n_steps=20,
        )

    else:

        trainer = Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=args.n_epochs,
            logger=loggers,
            callbacks=checkpoint_callbacks,
            log_every_n_steps=20,
        )

    torch.set_float32_matmul_precision('medium')

    # Train model
    trainer.fit(model, dm)

if __name__ == "__main__":
    main()