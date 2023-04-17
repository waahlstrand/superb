from models import SuperbModel, CustomSuperbBackbone
from augmentations import Augmenter
from superb import BinaryDataset, CategoricalDataset
from torch.utils.data import Subset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision import models
import json
import argparse
import warnings
import time
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
    NAME                = 'superb'
    SEVERITY            = 0

    parser.add_argument('--source', type=str, default='~/data/balder/datasets/superb/patients')
    parser.add_argument('--cfg', type=str, nargs='+', default=CONFIG_PATH)
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



    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Human readable time
    name = time.strftime("%Y%m%d-%H%M%S")

    # Read config
    with open(args.cfg, 'r') as f:
        config = json.load(f)

    DATA_ROOT               = Path(args.source)
    REMOVED                 = config["removed"]
    RESIZE_SHAPE            = config["min_shape"]
    CLASS_DISTRIBUTION      = config["class_distribution"][args.label_type]

    # Set up logging
    # Check if logging dir exists
    if not Path(args.log_dir).exists():
        Path(args.log_dir).mkdir(parents=True)
    
    csv_logger = CSVLogger(args.log_dir, name=args.name)
    tb_logger  = TensorBoardLogger(args.log_dir, name=args.name)
    model_dir  = csv_logger.log_dir + "/checkpoints"
    checkpoint = ModelCheckpoint(model_dir, monitor='val_loss', save_top_k=2, mode='min')

    # Load data
    if args.label_type == 'binary':
        dataset     = BinaryDataset(DATA_ROOT, RESIZE_SHAPE, REMOVED, CLASS_DISTRIBUTION, severity=args.severity, mode='severity')
        n_classes   = 1
    elif args.label_type == 'multilabel':
        dataset = CategoricalDataset(DATA_ROOT, RESIZE_SHAPE, REMOVED, CLASS_DISTRIBUTION)
        n_classes = len(config["class_distribution"][args.label_type].keys())
    else:
        raise ValueError('Label type not recognized')
    
    # Undersample dataset
    idxs = [i for i, (_, y, _) in enumerate(dataset) if torch.minimum(y, torch.tensor(1)).item() == 1]
    jdxs = [i for i, (_, y, _) in enumerate(dataset) if y == 0]

    # Randomly sample from the majority class
    np.random.seed(args.seed)
    jdxs = np.random.choice(jdxs, len(idxs), replace=False)

    all_idxs = np.concatenate((idxs, jdxs))
    dataset = Subset(dataset, all_idxs)
    
    # Split data to ensure balanced classes in train and validation sets
    # idxs = np.arange(len(dataset))

    # ys   = np.array([y for _, y, _ in dataset])
    # sss = StratifiedShuffleSplit(
    #     n_splits=1, 
    #     test_size=1-args.train_fraction, 
    #     # test_size=dataset.class_distribution[0],
    #     random_state=args.seed
    #     )
    # train_idx, val_idx = next(sss.split(idxs, ys))

    # train_dataset       = Subset(dataset, train_idx)
    # validation_dataset  = Subset(dataset, val_idx)

    #

    ys   = np.array([y for _, y, _ in dataset])
    sss = StratifiedShuffleSplit(
        n_splits=1, 
        test_size=1-args.train_fraction, 
        # test_size=dataset.class_distribution[0],
        random_state=args.seed
        )
    train_idx, val_idx = next(sss.split(all_idxs, ys))

    train_dataset       = Subset(dataset, train_idx)
    validation_dataset  = Subset(dataset, val_idx)

    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(validation_dataset)}")
    

    train_dataloader    = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True)
    val_dataloader      = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)

    # Define model
    augmenter   = Augmenter(p=0.5)
    # backbone    = models.resnet50(pretrained=False, num_classes=n_classes)
    # backbone    = models.resnet50(pretrained=True, num_classes=1000)
    # for param in backbone.parameters():
    #     param.requires_grad = False

    # out_features = backbone.fc.in_features
    
    # backbone.fc = nn.Sequential(
    #     nn.Linear(out_features, n_classes),
    #     # nn.ReLU(inplace=True),
    #     # nn.Linear(out_features // 2, n_classes)
    # )

    backbone = CustomSuperbBackbone(n_classes)    

    # optimizer   = torch.optim.Adam(backbone.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer   = torch.optim.SGD(backbone.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    model       = SuperbModel(backbone, augmenter, optimizer, n_classes)
    trainer     = pl.Trainer(accelerator=args.device, max_epochs=args.n_epochs, logger=[tb_logger, csv_logger], callbacks=[checkpoint], log_every_n_steps=5)

    # Train model
    torch.set_float32_matmul_precision('medium')
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.save_checkpoint(f"{model_dir}/{name}_final.ckpt")


    # runtime = end - start

    # return runtime

if __name__ == "__main__":

    # start = time.time()
    # try:
    #     main()
    # except Exception as e:
    #     print(f"\n Exited early. Runtime: {time.time() - start}")
    #     print(e)
        
    # end = time.time()
    
    # print(f"\n Runtime: {end - start}")
    main()