from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from copy import deepcopy
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from distutils.util import strtobool

from utils.model_utils import ICTransformer

from ours.test import smapeofa, rmse, convert_tsf_to_dataframe, M4ZeroShotDataset, sliding_window_view, rolling_window, TopInd, BenchZeroShotDataset, testM4

from copy import deepcopy
PATH = '/your/path/to/data' 
CH_PATH =  '/your/path/to/checkpoint'
# 8640, 11520, 14400 = 12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24
# 34560, 46080, 57600 = 12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4

SEQ_LEN = 250
configs = { 
    'Weather': {'horizon': 96, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/weather/weather.csv', 'borders': None, 'max_size':500}, 
    'Traffic': {'horizon': 96, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/traffic/traffic.csv', 'borders': None, 'max_size':500}, 
    'Electricity': {'horizon': 96, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/electricity/electricity.csv', 'borders': None, 'max_size':500}, 
    'ILI': {'horizon': 24, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/illness/national_illness.csv', 'borders': None, 'max_size':500}, 
    'ETTh1': {'horizon': 96, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/ETT-small/ETTh1.csv', 'borders': [8640, 11520, 14400], 'max_size':500}, 
    'ETTh2': {'horizon': 96, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/ETT-small/ETTh2.csv', 'borders': [8640, 11520, 14400], 'max_size':500}, 
    'ETTm1': {'horizon': 96, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/ETT-small/ETTm1.csv', 'borders': [34560, 46080, 57600], 'max_size':500}, 
    'ETTm2': {'horizon': 96, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/ETT-small/ETTm2.csv', 'borders': [34560, 46080, 57600], 'max_size':500}, 


}

device = 'cuda:0'
model = ICTransformer().to(device)

checkpoint = torch.load(CH_PATH)
model.load_state_dict(checkpoint['model'])
model.eval()
maps = {'Weather': 24, 'Traffic': 168, 'Electricity': 24, 'ILI': 52, 'ETTh1': 12, 'ETTh2': 12, 'ETTm1': 48, 'ETTm2': 48}
metrics_bench = pd.DataFrame()
metrics_bench = pd.read_csv('results_fewshot.csv', index_col='Unnamed: 0')

tops = {}

for n_h, horizon in enumerate([96, 192, 336, 720]):
    for history_size_coef in [1, 2, 3, 4, 5, 6, 7, 8]: 
        for name, config in configs.items():
            config = deepcopy(config)
            config['horizon'] = horizon if name != 'ILI' else int(12 * (n_h + 2))
            config['real_size'] = int(maps[name] * (2 ** history_size_coef)) if name != 'ILI' else int(maps[name] * history_size_coef)
            if config['real_size'] >= 500:
                config['real_size'] = 490
            name = deepcopy(name) + f"_{config['horizon']}_{config['real_size']}"
            if name not in list(metrics_bench.columns):
                m4dataset = BenchZeroShotDataset(**config)
                loader = DataLoader(m4dataset, batch_size=512, shuffle=False, drop_last=False, num_workers=8)
                preds, trues = testM4(model, loader, device)
                preds, trues = trues.flatten(), preds.flatten()
                mae_i, mse_i = mean_absolute_error(trues, preds), mean_squared_error(trues, preds)
                metrics_bench.loc['series', name] = len(trues) // len(m4dataset.targets)
                metrics_bench.loc['horizon', name] = config['horizon']
                metrics_bench.loc['real_size', name] = config['real_size']
                metrics_bench.loc['coef', name] = history_size_coef
                metrics_bench.loc['mse', name] = mse_i
                metrics_bench.loc['mae', name] = mae_i


        print(metrics_bench.round(3))
        metrics_bench.to_csv('results_fewshot_exp7new_base_exp.csv')
