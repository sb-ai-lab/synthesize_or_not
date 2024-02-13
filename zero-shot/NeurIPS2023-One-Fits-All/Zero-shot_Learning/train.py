from data_provider.data_factory import data_provider
from utils.tools_tsf import EarlyStopping, adjust_learning_rate, visual, vali, test
from tqdm import tqdm
from models.GPT4TS import GPT4TS
from models.PatchTST import PatchTST
from models.DLinear import DLinear

import numpy as np
import torch
import torch.nn as nn

import os
import time

import warnings
import numpy as np

import argparse
import random

warnings.filterwarnings('ignore')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Zero-shot Train')

parser.add_argument('--model', type=str, default='PatchTST_M3_Yearly', help='model full name')
parser.add_argument('--config_path', type=str, default='./configs/', help='path to config file with models_config.py and datasets_config.py')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='path to checkpoints directory')
args = parser.parse_args()

# import config classes
import sys
sys.path.append(args.config_path)
from models_config import ModelsFactory

model_params = ModelsFactory()[args.model]
dataset_params = model_params.source_dataset_params()

path = os.path.join(args.checkpoints, f'{args.model}')
os.makedirs(path, exist_ok=True)

best_model_path = path + '/' + 'checkpoint.pth'    

if dataset_params.freq == 0:
    dataset_params.freq = 'h'

args.model = model_params.model

# data params
args.source_scaler=None
args.data = dataset_params.dataset_type
args.root_path = dataset_params.root_path
args.data_path = dataset_params.data_path
args.seq_len = model_params.seq_len
args.pred_len = dataset_params.horizon
args.label_len = model_params.label_len
args.batch_size = model_params.batch_size
args.season = dataset_params.season
args.freq = model_params.freq = dataset_params.freq
args.embed = model_params.embed = dataset_params.embed
args.percent = model_params.percent
args.max_len = model_params.max_len
args.features = dataset_params.features
args.target = dataset_params.target
args.num_workers = dataset_params.num_workers

# train params
args.decay_fac = model_params.decay_fac
args.lradj = model_params.lradj
args.learning_rate = model_params.learning_rate

train_data, train_loader = data_provider(args, 'train', train_all=model_params.train_all)

args.batch_size = model_params.test_batch_size
vali_data, vali_loader = data_provider(args, 'val', drop_last_test=False, train_all=model_params.train_all)
test_data, test_loader = data_provider(args, 'test', drop_last_test=False)

args.freq = args.season

device = torch.device('cuda:0')

time_now = time.time()
train_steps = len(train_loader)

model_params.pred_len = dataset_params.horizon

if args.model == 'PatchTST':
    model = PatchTST(model_params, device)
    model.to(device)
elif args.model == 'DLinear':
    model = DLinear(model_params, device)
    model.to(device)
else:
    model = GPT4TS(model_params, device)
    model.to(device)
params = model.parameters()
model_optim = torch.optim.Adam(params, lr=model_params.learning_rate)

early_stopping = EarlyStopping(patience=400, verbose=True)
if model_params.loss_func == 'mse':
    criterion = nn.MSELoss()

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=model_params.tmax, eta_min=1e-8)

for epoch in range(model_params.train_epochs):
    iter_count = 0
    train_loss = []

    epoch_time = time.time()
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
        
        iter_count += 1
        model_optim.zero_grad()
        batch_x = batch_x.float().to(device)

        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)
        
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

        outputs = model(batch_x, batch_x_mark)

        outputs = outputs[:, -args.pred_len:, :]
        batch_y = batch_y[:, -args.pred_len:, :].to(device)
        loss = criterion(outputs, batch_y)
        train_loss.append(loss.item())

        loss.backward()
        model_optim.step()

        if (i + 1) % model_params.print_int == 0:
            print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
            speed = (time.time() - time_now) / iter_count
            left_time = speed * ((model_params.train_epochs - epoch) * train_steps - i)
            print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
            iter_count = 0
            time_now = time.time()

    print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

    train_loss = np.average(train_loss)
    vali_loss = vali(model, vali_data, vali_loader, criterion, args, device, None, training=True)

    print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
        epoch + 1, train_steps, train_loss, vali_loss))

    early_stopping(vali_loss, model, path, model_params)
    if model_params.cos:
        scheduler.step()
        print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
    else:
        adjust_learning_rate(model_optim, epoch + 1, args)
    if early_stopping.early_stop:
        print("Early stopping")
        break
