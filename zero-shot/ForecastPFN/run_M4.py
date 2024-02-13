from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

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
from keras import backend
from einops import rearrange
import time
import os
from dateutil.relativedelta import relativedelta

# from data_provider.data_factory import data_provider
# from ..utils.metrics import smape

def smape(y_true, y_pred):
    """ Calculate Armstrong's original definition of sMAPE between `y_true` & `y_pred`.
        `loss = 200 * mean(abs((y_true - y_pred) / (y_true + y_pred), axis=-1)`
        Args:
        y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
        Returns:
        Symmetric mean absolute percentage error values. shape = `[batch_size, d0, ..
        dN-1]`.
        """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    diff = tf.abs(
        (y_true - y_pred) /
        backend.maximum(y_true + y_pred, backend.epsilon())
    )
    return 200.0 * backend.mean(diff, axis=-1)

def smapeofa(true, pred, agg=np.mean):
    smape = 200 * np.abs(pred - true) / (np.abs(pred) + np.abs(true) + 1e-8)
    if agg is None:
        return smape
    else:
        return agg(smape)
    
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)
        loaded_data.to_csv('testm.csv')

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )

class M4ZeroShotDataset(Dataset):   
    def __init__(self, train, horizon, history, test=None, max_size=500):
        self.make_test = False
        if test is None:
            self.make_test = True
        else:
            self.test = self.load_data(test)

        self.train, self.year, self.month, self.day, self.weekday, self.dayofyear = self.load_data(train)

        self.horizon = horizon
        self.history = history
        self.max_size = max_size


    def load_data(self, data):
        if isinstance(data, list):
            timeseries = data
        else:
            if isinstance(data, str):
                if data.split('.')[-1] == 'tsf':
                    data = convert_tsf_to_dataframe(data)[0]
#                     data.dropna(inplace=True)
                    timeseries = [self.dropna(ts).astype(np.float32) for ts in data.loc[:, 'series_value'].values]
#                     print(len(timeseries[0]))
#                     ts = 
                else:
                    data = pd.read_csv(data) 
                    data.dropna(inplace=True)
                    df1 = data.groupby('series_name')['series_value'].apply(list).reset_index(name='series_value_grouped')
                    timeseries = [ts for ts in df1.loc[:, 'series_value_grouped']]
                    df2 = data.groupby('series_name')['year'].apply(list).reset_index(name='true_year_grouped')
                    year = [ts for ts in df2.loc[:, 'true_year_grouped']]
                    df3 = data.groupby('series_name')['month'].apply(list).reset_index(name='true_month_grouped')
                    month = [ts for ts in df3.loc[:, 'true_month_grouped']]
                    df4 = data.groupby('series_name')['day'].apply(list).reset_index(name='true_day_grouped')
                    day = [ts for ts in df4.loc[:, 'true_day_grouped']]
                    df5 = data.groupby('series_name')['weekay'].apply(list).reset_index(name='true_weekday_grouped')
                    weekday = [ts for ts in df5.loc[:, 'true_weekday_grouped']]
                    df6 = data.groupby('series_name')['dayofyear'].apply(list).reset_index(name='true_dayofyear_grouped')
                    dayofyear = [ts for ts in df6.loc[:, 'true_dayofyear_grouped']]
                    
#                     years = [ts.values.astype(np.int) for n, ts in data.iloc[:, 1:].iterrows()]
#                     timeseries = [self.dropna(ts).values.astype(np.float32) for n, ts in data.iloc[:, 1:].iterrows()]
        return timeseries, year, month, day, weekday, dayofyear


    @staticmethod
    def dropna(x):
        return x[~np.isnan(x)]

    def __len__(self):
        return len(self.train)

    def __getitem__(self, index): 
        year = self.year[index]
        month = self.month[index]
        day = self.day[index]
        weekday = self.weekday[index]
        dayofyear = self.dayofyear[index]
        series_train = self.train[index]
        
        if self.make_test:
            series_test = series_train[-self.horizon:]
            series_train = series_train[:-self.horizon]
        else:
            series_test = self.test[index]

        len_train = np.minimum(self.history, len(series_train))
        pad_size = self.max_size - len_train

        x_train = series_train[-len_train:]
        x_test = np.array(series_test, dtype='float32')
        len_test = np.minimum(self.horizon, len(series_test))
        
        
        train_year = year[-(len_train+len_test):-len_test]
        train_month = month[-(len_train+len_test):-len_test]
        train_day = day[-(len_train+len_test):-len_test]
        train_weekday = weekday[-(len_train+len_test):-len_test]
        train_dayofyear = dayofyear[-(len_train+len_test):-len_test]
        seq_x_mark = np.stack([train_year, train_month, train_day, train_weekday, train_dayofyear], axis=-1)
        
        test_year = year[-len_test:]
        test_month = month[-len_test:]
        test_day = day[-len_test:]
        test_weekday = weekday[-len_test:]
        test_dayofyear = dayofyear[-len_test:]
        seq_y_mark = np.stack([test_year, test_month, test_day, test_weekday, test_dayofyear], axis=-1)

#         train_min = np.min(x_train)
#         train_ptp = np.ptp(x_train)
#         if np.isclose(train_ptp, 0):
#             train_ptp = 1
#         x_train = (x_train - train_min) / train_ptp
#         x_val = (x_val - train_min) / train_ptp
#         x_train = x_train.astype(np.float32)
#         x_val = x_val.astype(np.float32)

#         mask_train = np.pad(np.ones(len(x_train)), (0, pad_size), mode = 'constant')
#         x_train = np.pad(x_train, (0, pad_size), mode = 'constant')
        
#         from forecastpfn
        if np.all(x_train == x_train[0]):
            x_train[-1] += 1
        x_train = np.expand_dims(x_train, axis=1)
        history = x_train.copy()
        mean = np.nanmean(history)
        std = np.nanstd(history)
        history = (history - mean) / std
        history_mean = np.nanmean(history[-6:])
        history_std = np.nanstd(history[-6:])
        local_scale = (history_mean + history_std + 1e-4)
        history = np.clip(history / local_scale, a_min=0, a_max=1)
        
        if x_train.shape[0] != 100:
            seq_x_mark = seq_x_mark.astype(np.int64)
            if x_train.shape[0] > 100:
                target = seq_x_mark[-100:, :]
#                 print(22, target.shape)
                history = history[-100:, :].astype(np.float32)
#                 print(23, history.shape)
            else:
#                 print('before pad', x_train.shape[0])
#                 print(x_train)
#                 print('before pad', seq_x_mark.shape)
#                 print(seq_x_mark.shape)
#                 print(x_train.shape[0])
                target = np.pad(seq_x_mark, pad_width=((100-x_train.shape[0], 0), (0, 0)), mode='constant')
#                 target = np.pad(seq_x_mark, (0, 100-x_train.shape[0]))
#                 print(24, target.shape)
#                 print('before pad', history.shape)
                history = np.pad(history, pad_width=((100-x_train.shape[0], 0), (0, 0)), mode='constant').astype(np.float32)
#                 history = np.pad(history, (0, 100-x_train.shape[0]))
#                 print(25, history.shape)
                
            history = np.repeat(np.expand_dims(history, axis=0), self.horizon, axis=0)[:, :, 0].astype(np.float32) #убрать expand_dims
#             history = np.repeat(history, self.horizon, axis=0)[:, 0]
#             history = np.expand_dims(history, axis=0)
#             print('aaa', history.shape)
#             print(history)
#             ts = target # 
#             print(target.shape)
            ts = np.repeat(np.expand_dims(target, axis=0), self.horizon, axis=0)
    
        else:
            ts = np.repeat(np.expand_dims(seq_x_mark, axis=0), self.horizon, axis=1).astype(np.int64)
#             ts = seq_x_mark.astype(np.int64)
#             print(27, ts.shape)
            history = history.astype(np.float32)
#             print(28, history.shape)

#         print(ts.shape)
            
        task = np.full((self.horizon, ), 1).astype(np.int32)
#         task = 1
#         task = task.astype(np.int64)
#         task = np.int64(1)
#         task = 1
#         target_ts = np.expand_dims(seq_y_mark[-self.horizon:, :], axis=1).astype(np.int64)
#         print(seq_y_mark.shape)
#         target_ts = seq_y_mark[-self.horizon:, :].astype(np.int64)
#         target_ts = seq_y_mark.astype(np.int64)
        target_ts = np.expand_dims(seq_y_mark[-self.horizon:, :], axis=1).astype(np.int64)
#         print(29, target_ts.shape)
#         print(seq_y_mark.shape)
#         print('ts', ts.shape)
#         print('history', history.shape)
#         print('target_ts', target_ts.shape)
#         print('task', task.shape)
        model_input = {'ts': ts, 'history': history, 'target_ts': target_ts, 'task': task}
#         model_input = {'history': history, 'task': task}
#         print(history.shape)

        return model_input, mean, std, x_test

def test(model, loader, device):
    trues = []
    preds = []
    with torch.no_grad():
        for num, batch in enumerate(tqdm(loader)):
            model_input = batch[0]
#             {key: value for key, value in model_input.items(): einsum(value)}
#             print(model_input['ts'].shape)
            model_input.update((k, rearrange(v, 'b h ... -> (b h) ...')) for k,v in model_input.items())
            mean = batch[1]
            std = batch[2]
            true_vals = batch[3].numpy()
            pred_vals = model(model_input) #.data.cpu()
            scaled_vals = pred_vals['result'].cpu().numpy().T.reshape(-1) * pred_vals['scale'].cpu().numpy().reshape(-1)
#             print(pred_vals['scale'])
#             print(pred_vals['result'])
#             print(scaled_vals.shape)
            scaled_vals = rearrange(scaled_vals, '(b h) -> b h', b=loader.dataset.horizon)
#             print(std.shape)
#             print(scaled_vals.shape)
            scaled_vals_ = scaled_vals.copy()
#             std = torch.unsqueeze(std, dim=-1).repeat(1, loader.dataset.horizon).numpy()
#             mean = torch.unsqueeze(mean, dim=-1).repeat(1, loader.dataset.horizon).numpy()
            std = np.tile(np.expand_dims(std.numpy(), axis=-1), (1, loader.dataset.horizon))
            mean = np.tile(np.expand_dims(mean.numpy(), axis=-1), (1, loader.dataset.horizon))
            scaled_vals = std * scaled_vals.T + mean
#             print(scaled_vals.shape)
#             print(true_vals.shape)

            trues.append(list(true_vals))
            preds.append(list(scaled_vals))
        
#         обратно в rearrange
              
    trues = np.vstack(trues)
    preds = np.vstack(preds)
#     mae = np.abs(np.flatten())
#     return preds, trues, scaled_vals, std, mean
    return trues, preds # true_vals, scaled_vals, std, mean, pred_vals

if __name__ == '__main__':
    config = {}
    PATH = 'academic_data'
    
# #     horizons
    horizons =  [6] # [720] # [96, 192, 336, 720]
    history = 36
    for horizon in horizons:
        config[f'M4Year_{horizon}_{history}'] = {
            'horizon': horizon, 
            'history': history, 
            'train': 'm4_year.csv', 
            'max_size': 500,
            'flag': 'zero-shot'
        }
        config[f'M4Month_{horizon}_{history}'] = {
            'horizon': horizon, 
            'history': history, 
            'train': 'm4_month.csv', 
            'max_size': 500,
            'flag': 'zero-shot'
        }
        config[f'M4Quarter_{horizon}_{history}'] = {
            'horizon': horizon, 
            'history': history, 
            'train': 'm4_quarter.csv', 
            'max_size': 500,
            'flag': 'zero-shot'
        }
        config[f'M4Week_{horizon}_{history}'] = {
            'horizon': horizon, 
            'history': history, 
            'train': 'm4_week.csv', 
            'max_size': 500,
            'flag': 'zero-shot'
        }
        config[f'M4Day_{horizon}_{history}'] = {
            'horizon': horizon, 
            'history': history, 
            'train': 'm4_day.csv', 
            'max_size': 500,
            'flag': 'zero-shot'
        }
        config[f'M4Hour_{horizon}_{history}'] = {
            'horizon': horizon, 
            'history': history, 
            'train': 'm4_hour.csv', 
            'max_size': 500,
            'flag': 'zero-shot'
        }
        
    mse_list = []
    mae_list = []
    smape_list = []
    nan_ratio_list = []
    nan_count_list = []
    name_list = []
        
    for name, conf in config.items():
        start_time = time.time()
        device = 'cuda'
#         print({k: conf[k] for k in set(list(conf.keys())) - set('flag')})
        dataset = M4ZeroShotDataset(**{k: conf[k] for k in set(list(conf.keys())) - set({'flag'})})
        loader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=False,
            num_workers=0,
            drop_last=False
        )
        print(len(loader))
        print(conf)
        pretrained = tf.keras.models.load_model("saved_weights/", custom_objects={'smape': smape})
        t = test(pretrained, loader, device)
        print("--- %s seconds ---" % (time.time() - start_time))
        true = t[0]
        preds = t[1]
        nan_ratio = (len(preds.flatten()) - np.count_nonzero(np.isnan(preds))) / len(preds.flatten())
        nan_count = np.count_nonzero(np.isnan(preds))
        mae = np.nanmean(np.abs(true.flatten() - preds.flatten()))
        mse = np.nanmean(((true.flatten() - preds.flatten())**2))
        smape = np.nanmean(200 * np.abs(preds.flatten() - true.flatten()) / (np.abs(preds.flatten()) + np.abs(true.flatten()) + 1e-8))
        print(f'{name}')
        print(f'mse: {mse}')
        print(f'mae: {mae}')
        print(f'smape: {smape}')
        print(f'nan_ratio: {nan_ratio}')
        print(f'nan_count: {nan_count}')
        mse_list.append(mse)
        mae_list.append(mae)
        smape_list.append(smape)
        nan_ratio_list.append(nan_ratio)
        nan_count_list.append(nan_count)
        name_list.append(name)
           
    data = {'mse': mse_list, 'mae': mae_list, 'smape': smape_list, 'dataset': name_list, 'nan_ratio': nan_ratio_list, 'nan_count': nan_count_list}
        
    df = pd.DataFrame.from_dict(data)
    df.to_csv(f'exps/m4.csv')



