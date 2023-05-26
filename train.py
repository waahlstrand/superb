from models import CustomSuperbBackbone
from models.augmentations import Augmenter
from slask.superb import BinaryDataset, CategoricalDataset
from torch.utils.data import Subset, DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision import models
import json
import argparse
import warnings
import time
import matplotlib
import utils.labels as labelling

matplotlib.use('Agg')

warnings.filterwarnings("ignore")

def main():
    
        parser = argparse.ArgumentParser(description='Train a SUPERB model')

        CONFIG_PATH         = './configs/data.json'
        BATCH_SIZE          = 2
        N_WORKERS           = 8
        TRAIN_FRACTION      = 0.9
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

        args = parser.parse_args()

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # Human readable time
        name = time.strftime("%Y%m%d-%H%M%S")

        # Read config
        with open(args.cfg, 'r') as f:
                config = json.load(f)

        DATA_ROOT               = Path(args.source)
        REMOVED                 = config["removed"]
        RESIZE_SHAPE            = (300, 100)#config["min_shape"]

        dataset     = BinaryDataset(DATA_ROOT, RESIZE_SHAPE, REMOVED, severity=args.severity, mode='severity')
        n_classes   = 1
        # dataset     = CategoricalDataset(DATA_ROOT, RESIZE_SHAPE, REMOVED, severity=args.severity, mode='exists')
        # n_classes   = len(labelling.VERTEBRA_NAMES)

        # Undersample dataset
        idxs = [i for i, (_, y) in enumerate(dataset) if y > args.severity]
        jdxs = [i for i, (_, y) in enumerate(dataset) if y == 0]

        print(f"Number of positive samples: {len(idxs)}")
        print(f"Number of negative samples: {len(jdxs)}")

        # Randomly sample from the majority class
        jdxs = np.random.choice(jdxs, len(idxs), replace=False)

        all_idxs = np.concatenate((idxs, jdxs))
        subset = Subset(dataset, all_idxs)


        ys   = np.array([y for _, y in subset])
        sss = StratifiedShuffleSplit(
                n_splits=1, 
                test_size=1-args.train_fraction, 
                random_state=args.seed
                )
        
        train_idx, val_idx = next(sss.split(all_idxs, ys))

        train_dataset       = Subset(subset, train_idx)
        validation_dataset  = Subset(subset, val_idx)

        print(f"Train set size: {len(train_dataset)}")
        print(f"Validation set size: {len(validation_dataset)}")
        

        train_dataloader    = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True, collate_fn=dataset.collate_fn)
        val_dataloader      = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, collate_fn=dataset.collate_fn)

        # Define model
        train_augmenter   = Augmenter(p=0.5, mode='train')
        val_augmenter     = Augmenter(p=0, mode='val')

        # model = CustomSuperbBackbone(in_channels=8, n_classes=n_classes, n_layers=3)
        model = models.resnet152(pretrained=False, num_classes=n_classes)

        # Define optimizer
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # Make training loop
        criterion = nn.BCEWithLogitsLoss()
        # criterion = nn.NLLLoss()

        # Define device
        if args.device == 'gpu':
                device = torch.device('cuda:0')
        else:
                device = torch.device('cpu')

        model.to(device)

        # Define logging
        log_dir = Path(args.log_dir)
        log_dir.mkdir(exist_ok=True)

        log_file = log_dir / f'{args.name}_{name}.txt'

        with open(log_file, 'w') as f:
                f.write(f"Train set size: {len(train_dataset)}\n")
                f.write(f"Validation set size: {len(validation_dataset)}\n")
                f.write(f"Number of positive samples: {len(idxs)}\n")
                f.write(f"Number of negative samples: {len(jdxs)}\n")

        # Train model in loop
        for epoch in range(args.n_epochs):
                        
                        # Train
                        train_loss = train(model, train_dataloader, optimizer, criterion, device, train_augmenter, epoch, log_file)
        
                        # Validate
                        val_loss, val_acc = validate(model, val_dataloader, criterion, device, val_augmenter, epoch, log_file)
        
                        # Save model
                        if epoch % 10 == 0:
                                model_path = log_dir / f'{args.name}_{name}_{epoch}.pt'
                                torch.save(model.state_dict(), model_path)
        

def train(model, train_dataloader, optimizer, criterion, device, augmenter, epoch, log_file):
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_dataloader):

                data = augmenter(data)
                # print(data, data.min(), data.max())
                # plt.hist(data.cpu().flatten())
                # plt.savefig(f"hist_{batch_idx}.png")
                # print(data.shape)

                target = torch.tensor(target, dtype=torch.float32)

                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                output = output.reshape_as(target)
                target_binary = torch.minimum(target, torch.ones_like(target))
                # print(output.shape, target_binary.shape)
                # print(output, target_binary)
                loss = criterion(output, target_binary)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                if batch_idx % 10 == 0:
                        print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_dataloader.dataset)} ({100. * batch_idx / len(train_dataloader):.0f}%)]\tLoss: {loss.item():.6f}')
                        with open(log_file, 'a') as f:
                                f.write(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_dataloader.dataset)} ({100. * batch_idx / len(train_dataloader):.0f}%)]\tLoss: {loss.item():.6f}\n')

                if loss.item() < 0:
                        print('Loss is negative')
                        print(f"Data characteristics: {data.shape}, {data.dtype}, {data.min()}, {data.max()}")
                        print(f"Target characteristics: {target.shape}, {target.dtype}, {target.min()}, {target.max()}")
                        print(f"Output characteristics: {output.shape}, {output.dtype}, {output.min()}, {output.max()}")

        train_loss /= len(train_dataloader.dataset)
        print(f'====> Epoch: {epoch} Average loss: {train_loss:.4f}')
        with open(log_file, 'a') as f:
                f.write(f'====> Epoch: {epoch} Average loss: {train_loss:.4f}\n')
        return train_loss

def validate(model, val_dataloader, criterion, device, augmenter, epoch, log_file):
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
                for batch_idx, (data, target) in enumerate(val_dataloader):	

                        data = augmenter(data)

                        # if batch_idx == 0:

                        #         for i in range(data.shape[0]):
                        #                 plt.imshow(data[i, 0, :, :].cpu().numpy())
                        #                 plt.savefig(f"val_img_{i}.png")
                        #                 plt.close()

                        target = torch.tensor(target, dtype=torch.float32)

                        data, target = data.to(device), target.to(device)

                        output = model(data)
                        output = output.reshape_as(target)
                        
                        target_binary = torch.minimum(target, torch.ones_like(target))
                        
                        
                        val_loss += criterion(output, target_binary).item()
                        pred = torch.sigmoid(output).round()
                        #print(pred, target_binary)
                        correct += pred.eq(target_binary.view_as(pred)).sum().item()
        
        val_loss /= len(val_dataloader.dataset)
        val_acc = 100. * correct / len(val_dataloader.dataset)
        print(f'====> Validation set loss: {val_loss:.4f}, accuracy: {val_acc:.0f}%')
        with open(log_file, 'a') as f:
                f.write(f'====> Validation set loss: {val_loss:.4f}, accuracy: {val_acc:.0f}%\n')
        return val_loss, val_acc

if __name__ == '__main__':
        main()
