from slask.new import SuperbModel, SimSiam, PatchwiseSimSiam
from data.superb import RSNA
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
from sklearn.neighbors import KNeighborsClassifier
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

    parser.add_argument('--source', type=str, default='/data/balder/datasets/rsna/stage_2_train_images')
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
    parser.add_argument('--patchwise', type=bool, default=False)


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
    
        tb_logger  = TensorBoardLogger(args.log_dir, name=args.name)

        loggers = [tb_logger]

        model_dir  = loggers[0].log_dir + "/checkpoints"
        checkpoint = ModelCheckpoint(model_dir, monitor='train_loss', save_top_k=2, mode='min')

        callbacks = [checkpoint]


    # Load data
    # Read config
    with open(args.cfg, 'r') as f:
        config = json.load(f)

    data_root       = Path(args.source) if args.source else Path(config['root'])
    labels_path     = Path(config['labels'])
    dataset         = RSNA(data_root, labels_path, size=(512, 512))

    # Get only negative samples
    # idxs = [i for i, (x, y) in enumerate(dataset) if y == 0]
    idxs = dataset.where_label(lambda x: x == 0)
    dataset = Subset(dataset, idxs)
    
    # Only select the negative examples
    dataloader    = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True)

    # Set up model
    if not args.patchwise:
        model = SimSiam()
    else:
        model = PatchwiseSimSiam(size=(128, 128), p=1.0, expansion_factor=0.5)

    # Set up trainer
    trainer = L.Trainer(
        accelerator=args.device,
        max_epochs=args.n_epochs,
        logger=loggers,
        callbacks=callbacks)
    
    # Train
    trainer.fit(model, dataloader)

    # Test with knn classifier
    rsna_data = RSNA(data_root, labels_path, size=(512, 512))
    # Get features
    features = []
    labels = []
    for i, (x, y) in tqdm(enumerate(rsna_data), total=len(rsna_data)):
        features.append(model.encoder(x.unsqueeze(0)).detach().cpu().numpy())
        labels.append(y)

    features = np.concatenate(features, axis=0)
    labels = np.array(labels)

    # Split into train and test
    train_idxs, test_idxs = train_test_split(np.arange(len(labels)), train_size=0.85, stratify=labels)

    # Train knn
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(features[train_idxs], labels[train_idxs])

    # Test knn
    acc = knn.score(features[test_idxs], labels[test_idxs])
    print(f"Accuracy: {acc}")





if __name__ == "__main__":

    main()