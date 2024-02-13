from data_provider.data_factory import data_provider
from utils.tools_tsf import test, convert_tsf_to_dataframe
from models.GPT4TS import GPT4TS
from models.PatchTST import PatchTST
from models.DLinear import DLinear
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer

import numpy as np
import pandas as pd
import torch

import os

import warnings

import argparse
import random

warnings.filterwarnings('ignore')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

TARGET_COLUMN_M3_M4 = "series_value"

def initialize_params(args, dataset_params, model_params):
    def _get_source_scaler(args, model_params):
        source_data = convert_tsf_to_dataframe(f"{model_params.source_dataset_params.root_path}/{model_params.source_dataset_params.data_path}")[0]
        if args.source_scaling == "standard_scaler":
            source_scaler = StandardScaler().fit(np.hstack(source_data[TARGET_COLUMN_M3_M4].values).reshape(-1, 1))
        elif args.source_scaling == "min_max_scaler":
            source_scaler = MinMaxScaler().fit(np.hstack(source_data[TARGET_COLUMN_M3_M4].values).reshape(-1, 1))
        elif args.source_scaling == "quantile_transformer":
            source_scaler = QuantileTransformer().fit(np.hstack(source_data[TARGET_COLUMN_M3_M4].values).reshape(-1, 1))
        else:
            return False
        return source_scaler
    args.data = dataset_params.dataset_type
    args.model = model_params.model
    args.root_path = dataset_params.root_path
    args.data_path = dataset_params.data_path
    args.seq_len = model_params.seq_len
    args.pred_len = model_params.pred_len
    args.target_pred_len = model_params.target_pred_len
    args.label_len = model_params.test_label_len
    args.batch_size = model_params.test_batch_size
    args.season = dataset_params.season
    args.embed = model_params.embed = dataset_params.embed
    args.freq = model_params.freq = dataset_params.freq
    args.percent = model_params.percent
    args.max_len = model_params.max_len
    args.features = dataset_params.features
    args.target = dataset_params.target
    args.num_workers = dataset_params.num_workers
    args.source_scaler = _get_source_scaler(args, model_params)
    return args


parser = argparse.ArgumentParser(description='Zero-shot Inference')

parser.add_argument('--model', type=str, default='DLinear_M3_Quarterly', help='model full name')
parser.add_argument('--config_path', type=str, default='./configs/', help='path to config file with models_config.py and datasets_config.py')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='path to checkpoints directory')
parser.add_argument('--target_data', type=str, default='M4_Yearly', help='test dataset name')
parser.add_argument('--res_path', type=str, default='./results/test.csv', help='path to save dataframe with mean metrics')
parser.add_argument('--source_scaling', type=str, default='False', help='one of ["standard_scaler", "min_max_scaler", "quantile_loss", False]')
parser.add_argument('--test_on_val', type=int, default=0, help='whether to use validation part of dataset to get metrics')
args = parser.parse_args()

if args.source_scaling == "False":
    args.source_scaling = False

# import config classes
import sys
sys.path.append(args.config_path)
from datasets_config import DatasetsFactory

mses = []
maes = []
smapes = []

path = os.path.join(args.checkpoints, f'{args.model}')
best_model_path = path + '/' + 'checkpoint.pth'

checkpoint = torch.load(best_model_path)

device = torch.device('cuda:0')
model_params = checkpoint["model_params"]
dataset_params = DatasetsFactory()[args.target_data]
model_params.pred_len = model_params.source_dataset_params.horizon
model_params.target_pred_len = min(model_params.source_dataset_params.horizon, dataset_params.target_horizon)

# For initialization of PatchTST we need these params in model_params
model_params.embed = dataset_params.embed
model_params.freq = dataset_params.freq

if model_params.model == 'PatchTST':
    model = PatchTST(model_params, device)
    model.to(device)
elif model_params.model == 'DLinear':
    model = DLinear(model_params, device)
    model.to(device)
else:
    model = GPT4TS(model_params, device)
    model.to(device)

model.load_state_dict(checkpoint["model_state_dict"])

args = initialize_params(args, dataset_params, model_params)

if args.test_on_val == 1:
    test_data, test_loader = data_provider(args, 'val', drop_last_test=False) 
else:
    test_data, test_loader = data_provider(args, 'test', drop_last_test=False)

args.freq = args.season

print(f"Dataset = {args.target_data}, Scaling = {args.source_scaling}")
print("Start inference")
mse, mae, smape, _, _ = test(model, test_data, test_loader, args, device, None)
mses.append(mse)
maes.append(mae)
smapes.append(smape)

mses = np.array(mses)
maes = np.array(maes)
smapes = np.array(smapes)

print("mse_mean = {:.4f}, mse_std = {:.4f}".format(np.mean(mses), np.std(mses)))
print("mae_mean = {:.4f}, mae_std = {:.4f}".format(np.mean(maes), np.std(maes)))
print("smapes_mean = {:.4f}, smapes_std = {:.4f}".format(np.mean(smapes), np.std(smapes)))

res_df = pd.DataFrame([{
    "mse_mean": np.mean(mses),
    "mae_mean": np.mean(maes),
    "smapes_mean": np.mean(smapes)
}])

res_df.to_csv(args.res_path, index=False)
