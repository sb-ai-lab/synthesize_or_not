import numpy as np
import pandas as pd
import os 
from tqdm import tqdm
import numba as nb
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import math
import joblib
import scipy.special
from datetime import datetime
from torch import Tensor
from typing import Union, Tuple, List, Dict
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import transformers
from transformers import get_scheduler, get_cosine_schedule_with_warmup
from time import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from utils.model_utils import MaskedRMSELoss, ICTransformer, MaskedSMAPELoss, MaskedMSELoss
from utils.plot_utils import plot_loss, plot_lr, plot_series
from utils.generate_utils import TimeSeriesDataset_trend
from utils.seed_utils import set_seed

from flags import parse_handle

def train_one_epoch(model, loss_fn, optimizer, scheduler, train_loader, device):
    lr_list = []
    loss_list = []

    for i, data in enumerate(train_loader):
        series = data['series'].to(device)
        mask_series = data['mask_series'].to(device)
        target = data['target'].to(device)
        mask_target = data['mask_target'].to(device)
        optimizer.zero_grad()
        outputs = model(series, mask_series)
        targets = target * mask_target
        outputs = outputs * mask_target
        loss = loss_fn(outputs, targets, mask_target) # / torch.sum(data['mask_target'].to(device)) * data['target'].to(device).shape[1] * data['target'].to(device).shape[0]
        lr_list.append(scheduler.get_lr())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        scheduler.step()
        loss_list.append(loss.item())
        print(loss)

    return loss_list, lr_list

def train(model, train_loader, val_loader, optimizer, scheduler, epochs, device):
    epoch_number = 0

    train_loss = []
    val_loss = []
    loss_fn = MaskedSMAPELoss()
    lr_list = []
    vloss_list = []


    for epoch in tqdm(range(epochs)):
        model.train()
        loss, lr_list_ = train_one_epoch(model, loss_fn, optimizer, scheduler, train_loader, device)
        lr_list.extend(lr_list_)
        vloss_list = []
        model.eval()
        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                series = vdata['series'].to(device)
                mask_series = vdata['mask_series'].to(device)
                target = vdata['target'].to(device)
                mask_target = vdata['mask_target'].to(device)
                voutputs = model(series, mask_series)
                vtargets = target * mask_target
                voutputs = voutputs * mask_target
                vloss = loss_fn(voutputs, vtargets, mask_target)
                vloss_list.append(vloss.item())
                print(vloss)

            # train_loss.extend(loss)
            # val_loss.extend(vloss_list)
            train_loss.append(np.mean(loss))
            val_loss.append(np.mean(vloss_list))


        if epoch % 2 == 0 or epoch == epochs - 1:
            with torch.no_grad():

                fig, axs = plt.subplots(2)
                axs[0].plot(train_loss, label='Train loss')
                axs[0].plot(val_loss, label='Val loss')
                axs[0].set_title(f'Train loss = {np.round(train_loss[-1], 5)}. Val loss = {np.round(val_loss[-1], 5)}.')
                plt.legend(fontsize="x-large")
                axs[1].plot(lr_list)
                axs[1].set_title('Learning rate')
                fig.tight_layout()
                lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
                lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
                fig.legend(lines, labels)
                plt.savefig(args.pics_loss_path + f'/losses_lr_{epoch}.png')
                plt.close()

                data = next(iter(train_loader))
                vdata = next(iter(val_loader))
                model.eval()
                pred = model(data['series'].to(device), data['mask_series'].to(device))
                vpred = model(vdata['series'].to(device), vdata['mask_series'].to(device))
                plot_series(data, pred, vdata, vpred, epoch, args.pics_series_path)

            checkpoint = { 
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            torch.save(checkpoint, args.model_path + f'/checkpoint_{epoch}.pth')

        epoch_number += 1

if __name__ == "__main__":
    set_seed(50)
    parser = parse_handle()
    args = parser.parse_args()

    if not os.path.exists(args.pics_series_path):
        os.makedirs(args.pics_series_path)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    if not os.path.exists(args.pics_loss_path):
        os.makedirs(args.pics_loss_path)

    torch.multiprocessing.set_sharing_strategy('file_system')
    EPOCHS = 400
    device = 'cuda'
    batch_size = 512
    PATH = '/path/to/checkpoint'
    model = ICTransformer().to(device)
    model.load_state_dict(torch.load(PATH)['model'])
    train_dataset = TimeSeriesDataset_trend(length=500000, offset = 1000001 * 3)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    val_dataset = TimeSeriesDataset_trend(length=100000, offset=1000001)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader) * 5, len(train_loader) * EPOCHS)
    train(model, train_loader, val_loader, optimizer, scheduler, EPOCHS, device)

