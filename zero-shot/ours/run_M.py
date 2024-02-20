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
CH_PATH = '/your/path/to/checkpoint'

SEQ_LEN = 250
#SEQ_LEN = 400

configs = { 
    'Weather_6_250': {'horizon': 6, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/weather/weather.csv', 'borders': None, 'max_size':500}, 
    'Traffic_6_250': {'horizon': 6, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/traffic/traffic.csv', 'borders': None, 'max_size':500}, 
    'Electricity_6_250': {'horizon': 6, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/electricity/electricity.csv', 'borders': None, 'max_size':500}, 
    'ILI_6_148': {'horizon': 6, 'history': 148, 'real_size': SEQ_LEN, 'train': f'{PATH}/illness/national_illness.csv', 'borders': None, 'max_size':500}, 
    'ETTh1_6_250': {'horizon': 6, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/ETT-small/ETTh1.csv', 'borders': [8640, 11520, 14400], 'max_size':500}, 
    'ETTh2_6_250': {'horizon': 6, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/ETT-small/ETTh2.csv', 'borders': [8640, 11520, 14400], 'max_size':500}, 
    'ETTm1_6_250': {'horizon': 6, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/ETT-small/ETTm1.csv', 'borders': [34560, 46080, 57600], 'max_size':500}, 
    'ETTm2_6_250': {'horizon': 6, 'history': 512, 'real_size': SEQ_LEN, 'train': f'{PATH}/ETT-small/ETTm2.csv', 'borders': [34560, 46080, 57600], 'max_size':500},


}
device = 'cuda:0'
model = ICTransformer().to(device)

checkpoint = torch.load(CH_PATH)
model.load_state_dict(checkpoint['model'])
model.eval()

metrics_bench = pd.DataFrame()

pred_ = []
true_ = []
for name, config in configs.items():
    if name not in list(metrics_bench.columns):
        m4dataset = BenchZeroShotDataset(**config)
        loader = DataLoader(m4dataset, batch_size=512, shuffle=False, drop_last=False, num_workers=8)
        preds, trues = testM4(model, loader, device)
        #preds = np.clip(preds, np.maximum(10, preds.min()), preds.max())
        preds, trues = trues.flatten(), preds.flatten()
        try:
            smape_i = smapeofa(trues.flatten(), preds.flatten(), np.mean)
        except:
            smape_i = None
        try:
            smape_med_i = smapeofa(trues.flatten(), preds.flatten(), np.median)
        except:
            smape_med_i = None
        try:
            mae_i = mean_absolute_error(trues.flatten(), preds.flatten())
        except:
            mae_i = None
        try:
            mse_i = mean_squared_error(trues.flatten(), preds.flatten())
        except:
            mse_i = None
        metrics_bench.loc['series', name] = len(trues) // len(m4dataset.targets)
        metrics_bench.loc['horizon', name] = config['horizon']
        metrics_bench.loc['smape', name] = smape_i
        #metrics_bench.loc['smape med', name] = smape_med_i
        metrics_bench.loc['mse', name] = mse_i
        metrics_bench.loc['mae', name] = mae_i


configsM4 = { 'Y': {'horizon': 6, 'history': 20, 'train': f'{PATH}/Yearly-train.csv', 'test':f'{PATH}/Yearly-test.csv', 'max_size':500}, 
            'Q': {'horizon': 6, 'history': 10, 'train': f'{PATH}/Quarterly-train.csv', 'test':f'{PATH}/Quarterly-test.csv', 'max_size':500}, 
            'M': {'horizon': 6, 'history': 10, 'train': f'{PATH}/Monthly-train.csv', 'test':f'{PATH}/Monthly-test.csv', 'max_size':500}, 
            'W': {'horizon': 6, 'history': 100, 'train': f'{PATH}/Weekly-train.csv', 'test':f'{PATH}/Weekly-test.csv', 'max_size':500}, 
            'D': {'horizon': 6, 'history': 10, 'train': f'{PATH}/Daily-train.csv', 'test':f'{PATH}/Daily-test.csv', 'max_size':500},
            'H': {'horizon': 6, 'history': 220, 'train': f'{PATH}/Hourly-train.csv', 'test':f'{PATH}/Hourly-test.csv', 'max_size':500},

}

for freq, config in configsM4.items():
    m4dataset = M4ZeroShotDataset(**config)
    loader = DataLoader(m4dataset, batch_size=512, shuffle=False, drop_last=False, num_workers=8)
    preds, trues = testM4(model, loader, device)
    try:
        smape_i = smapeofa(trues.flatten(), preds.flatten(), np.mean)
    except:
        smape_i = None
    try:
        smape_med_i = smapeofa(trues.flatten(), preds.flatten(), np.median)
    except:
        smape_med_i = None
    try:
        mae_i = mean_absolute_error(trues.flatten(), preds.flatten())
    except:
        mae_i = None
    try:
        mse_i = mean_squared_error(trues.flatten(), preds.flatten())
    except:
        mse_i = None
    metrics_bench.loc['series', freq] = len(trues) 
    metrics_bench.loc['horizon', freq] = config['horizon']
    metrics_bench.loc['smape', freq] = smape_i
    #metrics_bench.loc['smape med', name] = smape_med_i
    metrics_bench.loc['mse', freq] = mse_i
    metrics_bench.loc['mae', freq] = mae_i

metrics_bench.round(3)
metrics_bench.to_csv(f'results_zeroshot_M.csv')