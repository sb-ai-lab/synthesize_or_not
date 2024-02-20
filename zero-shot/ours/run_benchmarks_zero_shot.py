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
    'Weather_96_250': {'horizon': 96, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/weather/weather.csv', 'borders': None, 'max_size':500}, 
    'Traffic_96_250': {'horizon': 96, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/traffic/traffic.csv', 'borders': None, 'max_size':500}, 
    'Electricity_96_250': {'horizon': 96, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/electricity/electricity.csv', 'borders': None, 'max_size':500}, 
    'ILI_24_148': {'horizon': 24, 'history': 148, 'real_size': SEQ_LEN, 'train': f'{PATH}/illness/national_illness.csv', 'borders': None, 'max_size':500}, 
    'ETTh1_96_250': {'horizon': 96, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/ETT-small/ETTh1.csv', 'borders': [8640, 11520, 14400], 'max_size':500}, 
    'ETTh2_96_250': {'horizon': 96, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/ETT-small/ETTh2.csv', 'borders': [8640, 11520, 14400], 'max_size':500}, 
    'ETTm1_96_250': {'horizon': 96, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/ETT-small/ETTm1.csv', 'borders': [34560, 46080, 57600], 'max_size':500}, 
    'ETTm2_96_250': {'horizon': 96, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/ETT-small/ETTm2.csv', 'borders': [34560, 46080, 57600], 'max_size':500},
    'Weather_192_250': {'horizon': 192, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/weather/weather.csv', 'borders': None, 'max_size':500}, 
    'Traffic_192_250': {'horizon': 192, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/traffic/traffic.csv', 'borders': None, 'max_size':500}, 
    'Electricity_192_250': {'horizon': 192, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/electricity/electricity.csv', 'borders': None, 'max_size':500}, 
    'ILI_36_148': {'horizon': 36, 'history': 148, 'real_size': SEQ_LEN, 'train': f'{PATH}/illness/national_illness.csv', 'borders': None, 'max_size':500}, 
    'ETTh1_192_250': {'horizon': 192, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/ETT-small/ETTh1.csv', 'borders': [8640, 11520, 14400], 'max_size':500}, 
    'ETTh2_192_250': {'horizon': 192, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/ETT-small/ETTh2.csv', 'borders': [8640, 11520, 14400], 'max_size':500}, 
    'ETTm1_192_250': {'horizon': 192, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/ETT-small/ETTm1.csv', 'borders': [34560, 46080, 57600], 'max_size':500}, 
    'ETTm2_192_250': {'horizon': 192, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/ETT-small/ETTm2.csv', 'borders': [34560, 46080, 57600], 'max_size':500}, 
    'Weather_336_250': {'horizon': 336, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/weather/weather.csv', 'borders': None, 'max_size':500}, 
    'Traffic_336_250': {'horizon': 336, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/traffic/traffic.csv', 'borders': None, 'max_size':500}, 
    'Electricity_336_250': {'horizon': 336, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/electricity/electricity.csv', 'borders': None, 'max_size':500}, 
    'ILI_48_148': {'horizon': 48, 'history': 148, 'real_size': SEQ_LEN, 'train': f'{PATH}/illness/national_illness.csv', 'borders': None, 'max_size':500}, 
    'ETTh1_336_250': {'horizon': 336, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/ETT-small/ETTh1.csv', 'borders': [8640, 11520, 14400], 'max_size':500}, 
    'ETTh2_336_250': {'horizon': 336, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/ETT-small/ETTh2.csv', 'borders': [8640, 11520, 14400], 'max_size':500}, 
    'ETTm1_336_250': {'horizon': 336, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/ETT-small/ETTm1.csv', 'borders': [34560, 46080, 57600], 'max_size':500}, 
    'ETTm2_336_250': {'horizon': 336, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/ETT-small/ETTm2.csv', 'borders': [34560, 46080, 57600], 'max_size':500}, 
    'Weather_720_250': {'horizon': 720, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/weather/weather.csv', 'borders': None, 'max_size':500}, 
    'Traffic_720_250': {'horizon': 720, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/traffic/traffic.csv', 'borders': None, 'max_size':500}, 
    'Electricity_720_250': {'horizon': 720, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/electricity/electricity.csv', 'borders': None, 'max_size':500}, 
    'ILI_60_148': {'horizon': 60, 'history': 148, 'real_size': SEQ_LEN, 'train': f'{PATH}/illness/national_illness.csv', 'borders': None, 'max_size':500}, 
    'ETTh1_720_250': {'horizon': 720, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/ETT-small/ETTh1.csv', 'borders': [8640, 11520, 14400], 'max_size':500}, 
    'ETTh2_720_250': {'horizon': 720, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/ETT-small/ETTh2.csv', 'borders': [8640, 11520, 14400], 'max_size':500}, 
    'ETTm1_720_250': {'horizon': 720, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/ETT-small/ETTm1.csv', 'borders': [34560, 46080, 57600], 'max_size':500}, 
    'ETTm2_720_250': {'horizon': 720, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/ETT-small/ETTm2.csv', 'borders': [34560, 46080, 57600], 'max_size':500},  
   # 'Demand': {'horizon': 30, 'history': 50, 'real_size': SEQ_LEN, 'train': '/media/ssd-3t/Kostromina/pyboost_experiments/data/demand/demand_scaled_transformer_format.csv', 'borders': None, 'max_size':500, 'scale': False}, 


}

device = 'cuda:1'
model = ICTransformer().to(device)

checkpoint = torch.load(CH_PATH)
model.load_state_dict(checkpoint['model'])
model.eval()

#metrics_bench = pd.DataFrame()
metrics_bench = pd.read_csv('results_zeroshot.csv', index_col='Unnamed: 0')
tops = {}

pred_ = []
true_ = []
for name, config in configs.items():
    if name not in list(metrics_bench.columns):
        m4dataset = BenchZeroShotDataset(**config)
        loader = DataLoader(m4dataset, batch_size=512, shuffle=False, drop_last=False, num_workers=8)
        preds, trues = testM4(model, loader, device)
        #preds = np.clip(preds, np.maximum(10, preds.min()), preds.max())
        preds, trues = trues.flatten(), preds.flatten()
        mae_i, mse_i = mean_absolute_error(trues, preds), mean_squared_error(trues, preds)
        metrics_bench.loc['series', name] = len(trues) // len(m4dataset.targets)
        metrics_bench.loc['horizon', name] = config['horizon']
        #metrics_bench.loc['smape', name] = smape_i
        #metrics_bench.loc['smape med', name] = smape_med_i
        metrics_bench.loc['mse', name] = mse_i
        metrics_bench.loc['mae', name] = mae_i

    #tops[name] = np.argsort(smapeofa(trues, preds, None).mean(axis=1))
metrics_bench.to_csv('results_zeroshot.csv')