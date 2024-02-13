import numpy as np
from copy import deepcopy
import random
from torch.utils.data import Dataset

class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, series):
        l = len(series)

        mask = np.ones(l, np.float32)

        for n in range(self.n_holes):
            np.random.seed(12)
            y = np.random.randint(l)

            y1 = np.clip(y - self.length // 2, 0, l)
            y2 = np.clip(y + self.length // 2, 0, l)

            mask[y1: y2] = 0.
            
        mask = np.broadcast_to(mask, series.shape)
        series = series * mask

        return series
    

def generate_series(
    n_c = 5, #
    n_p = 10, #
    n = 13, #
    add_shift = True,
    seed1 = 1,
    seed2 = 2,
):
    # different seeds
    np.random.seed(seed1)
    cos = np.random.uniform(-2, 2, n_c)
    np.random.seed(seed2)
    sin = np.random.uniform(-2, 2, n_c)
    coefs = cos + 1j * sin
    series_ = list(np.fft.ifft(coefs, n, ).real) # [:n//2]
    first = series_[0]
    series = deepcopy(series_)
    series = np.tile(series_, n_p)

    if add_shift == True:
        pass

    return series, coefs

class TimeSeriesDataset(Dataset):   
    def __init__(self, offset=0, length=100000):
        # self.n_outputs = n_outputs
        self.max_len_train = 500 # round(0.25 * 25 * (25 + 1))
        self.max_len_target = 720 # round(0.75 * 25 * (32 + 1))
        self.all_length = self.max_len_train + self.max_len_target
        self.length = length
        self.offset = offset

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        seed = [index + self.offset + i for i in range(100)]
        np.random.seed(seed[0])
        n_c = np.random.randint(3, 8)
        np.random.seed(seed[1])
        n = np.random.randint(8, 200) 
        np.random.seed(seed[2])
        linear_coef = np.random.uniform(*random.choice([(-0.01, -0.0001), (0.0001, 0.01)]))
        trend_name = random.choice(2 * ['linear'] + ['log1'] + ['log'] + ['quadratic'] + 5 * [None] + ['exp'])
        np.random.seed(seed[3])
        log_coef = np.random.uniform(*random.choice([(-1, -0.01), (0.01, 1)]))
        np.random.seed(seed[4])
        log1_coef = np.random.uniform(*random.choice([(-1, -0.01), (0.01, 1)]))
        np.random.seed(seed[5])
        quadratic_coef = np.random.uniform(*random.choice([(-0.001, -0.01), (0.001, 0.01)]))
        quadratic_coef_sign = random.choice([-1, 1])
        np.random.seed(seed[6])
        exp_coef = np.random.uniform(*random.choice([(-0.005, -0.0005), (0.0005, 0.005)]))
        exp_coef_sign = random.choice([-1, 1])
        np.random.seed(seed[7])
        noise_std = np.random.uniform(0.001, 1)
        np.random.seed(seed[8])
        zeros_n_holes = np.random.randint(1, 10)
        np.random.seed(seed[9])
        zeros_length = np.random.randint(1, 10)
        np.random.seed(seed[10])
        start_train = np.random.randint(0, n)
        np.random.seed(seed[11])
        n_periods_train = np.random.randint(2, 8)
        np.random.seed(seed[12])
        n_random_points_train = np.random.randint(0, n)
        end_train = min(self.max_len_train, start_train + n_periods_train * n + n_random_points_train)
        np.random.seed(seed[13])
        n_p = np.random.randint(n_periods_train+2+5, 25)

        np.random.seed(seed[14])
        series, coefs = generate_series(
            n_c = n_c, #
            n_p = n_p, #
            n = n, #
            seed1 = index + 1 + self.offset,
            seed2 = index + 3 + self.offset,
        )
        np.random.seed(seed[15])
        noise = np.random.normal(0, noise_std, len(series))
        trend = add_trend(
            trend_name, 
            len(series), 
            linear_coef, 
            log1_coef, 
            log_coef, 
            quadratic_coef,
            exp_coef,
            quadratic_coef_sign,
            exp_coef_sign
        )
        np.random.seed(seed[16])
        clean_trend = np.random.choice(a=[False, True], p=[0.98, 1-0.98])
        train = series[start_train:end_train]
        target = series[end_train: (end_train + self.max_len_target)]
        train_min = np.min(train)
        train_ptp = np.ptp(train)

        # w/o trend
        train = (train - train_min) / train_ptp
        target = (target - train_min) / train_ptp

        if trend_name == None:
            trend_bool = False
        else:
            trend_bool = True

        # with trend
        # if clean_trend == True and trend_name == None:
        #     pass
        # else:    
        #     train = (train - train_min) / train_ptp
        #     target = (target - train_min) / train_ptp
        # if np.random.randint(1, 3) == 3:
        #     cutout = Cutout(**zeros_params)
        #     train = cutout(train)
            
        # mask_trend = np.pad(np.ones(len(trend)), (0, self.all_length - len(target) - len(train)), mode = 'constant')
        # trend = np.pad(trend, (0, self.all_length - len(train) - len(target)), mode = 'constant')
        # mask_noise = np.pad(np.ones(len(noise)), (0, self.all_length - len(target) - len(train)), mode = 'constant')
        # noise = np.pad(noise, (0, self.all_length - len(train) - len(target)), mode = 'constant')
        mask_target = np.pad(np.ones(len(target)), (0, self.max_len_target - len(target)), mode = 'constant')
        target = np.pad(target, (0, self.max_len_target - len(target)), mode = 'constant')
        mask_train = np.pad(np.ones(len(train)), (0, self.max_len_train - len(train)), mode = 'constant')
        train = np.pad(train, (0, self.max_len_train - len(train)), mode = 'constant')
        # mask_signal = np.pad(np.ones(len(series)), (0, self.all_length - len(target) - len(train)), mode = 'constant')
        # signal = np.pad(signal, (0, self.all_length - len(train) - len(target)), mode = 'constant')
        mask_fft_coefs = np.pad(np.ones(len(coefs)), (0, n_c), mode = 'constant')
        coefs = np.pad(coefs, (0, n_c), mode = 'constant')
        
        out = {
            # 'series_': series.astype('float32'),
            'series': train.astype('float32'), # (train.astype('float32') - 0.5) * 2,
            'target': target.astype('float32'), # (target.astype('float32') - 0.5) * 2,
            'mask_series': mask_train,
            'mask_target': mask_target,
            'trend_bool': trend_bool,
            # 'mask_trend': mask_trend,
            # 'trend': trend,
            # 'mask_noise': mask_noise,
            # 'noise': noise,
            # # 'mask_signal': mask_signal,
            # # 'signal': series,
            # 'mask_fft_coefs': mask_fft_coefs,
            # 'fft_coefs': coefs,
            'seed': seed,
        }
        return out

def add_trend(
    trend_name, 
    length, 
    linear_coef=None, 
    log1_coef=None, 
    log_coef=None, 
    quadratic_coef=None, 
    exp_coef=None,
    quadratic_coef_sign=None,
    exp_coef_sign=None,
):
    eps = 1e-10
    if trend_name == 'linear':
        trend = np.array([linear_coef * i for i in range(length)])
    elif trend_name == 'log1':
        trend = np.array([log1_coef * np.log1p(i + eps) for i in range(length)])
    elif trend_name == 'log':
        trend = np.array([log_coef * np.log(i + eps) for i in range(length)])
    elif trend_name  == 'quadratic':
        trend = np.array([quadratic_coef_sign * (i * quadratic_coef)**2 for i in range(length)])
    elif trend_name == 'exp':
        trend = np.array([exp_coef_sign * np.exp(exp_coef * i) for i in range(length)])
    elif trend_name  == None:
        trend = np.zeros(length)
    else:
        raise NotImplementedError
    return trend


class TimeSeriesDataset_trend(Dataset):   
    def __init__(self, offset=0, length=100000):
        # self.n_outputs = n_outputs
        self.max_len_train = 500 # round(0.25 * 25 * (25 + 1))
        self.max_len_target = 720 # round(0.75 * 25 * (32 + 1))
        self.all_length = self.max_len_train + self.max_len_target
        self.length = length
        self.offset = offset

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        seed = [index + self.offset + i for i in range(100)]
        np.random.seed(seed[0])
        n_c = np.random.randint(3, 8)
        np.random.seed(seed[1])
        n = np.random.randint(8, 200)
        trend_name = random.choice(2 * ['linear'] + ['log1'] + ['log'] + ['quadratic'] + 5 * [None] + ['exp'])

        np.random.seed(seed[2])
        if n > 100:
            linear_coef = np.random.uniform(*random.choice([(-1e-3, -1e-8), (1e-8, 1e-3)]))
        else:
            linear_coef = np.random.uniform(*random.choice([(-1e-3, -1e-6), (1e-6, 1e-3)]))
        # linear_coef = np.random.uniform(*random.choice([(-0.01, -0.0001), (0.0001, 0.01)]))
        
        np.random.seed(seed[3])
        if n > 100:
            log_coef = np.random.uniform(*random.choice([(-0.2, -0.001), (0.001, 0.2)]))
        else:
            log_coef = np.random.uniform(*random.choice([(-0.2, -0.01), (0.01, 0.2)]))
        # log_coef = np.random.uniform(*random.choice([(-1, -0.01), (0.01, 1)]))

        np.random.seed(seed[4])
        if n > 100:
            log1_coef = np.random.uniform(*random.choice([(-1, -0.0001), (0.0001, 1)]))
        else:
            log1_coef = np.random.uniform(*random.choice([(-1, -0.01), (0.01, 1)]))
        # log1_coef = np.random.uniform(*random.choice([(-1, -0.01), (0.01, 1)]))

        np.random.seed(seed[5])
        if n > 100:
            quadratic_coef = np.random.uniform(*random.choice([(-0.0007, -0.001), (0.001, 0.0007)]))
        else:
            quadratic_coef = np.random.uniform(*random.choice([(-0.007, -0.001), (0.001, 0.007)]))
        # quadratic_coef = np.random.uniform(*random.choice([(-0.007, -0.001), (0.001, 0.007)]))
        quadratic_coef_sign = random.choice([-1, 1])

        np.random.seed(seed[6])
        if n > 100:
            exp_coef = np.random.uniform(*random.choice([(-0.0005, -0.00005), (0.00005, 0.0005)]))
        else:
            exp_coef = np.random.uniform(*random.choice([(-0.005, -0.0005), (0.0005, 0.005)]))
        # exp_coef = np.random.uniform(*random.choice([(-0.005, -0.0005), (0.0005, 0.005)]))
        exp_coef_sign = random.choice([-1, 1])

        np.random.seed(seed[7])
        noise_std = np.random.uniform(0.001, 1)
        np.random.seed(seed[8])
        zeros_n_holes = np.random.randint(1, 10)
        np.random.seed(seed[9])
        zeros_length = np.random.randint(1, 10)
        np.random.seed(seed[10])
        start_train = np.random.randint(0, n)
        np.random.seed(seed[11])
        n_periods_train = np.random.randint(2, 8)
        np.random.seed(seed[12])
        n_random_points_train = np.random.randint(0, n)
        end_train = min(self.max_len_train, start_train + n_periods_train * n + n_random_points_train)
        np.random.seed(seed[13])
        n_p = np.random.randint(n_periods_train+2+5, 25)

        np.random.seed(seed[14])
        series, coefs = generate_series(
            n_c = n_c, #
            n_p = n_p, #
            n = n, #
            seed1 = index + 1 + self.offset,
            seed2 = index + 3 + self.offset,
        )
        np.random.seed(seed[15])
        noise = np.random.normal(0, noise_std, len(series))
        trend = add_trend(
            trend_name, 
            len(series), 
            linear_coef, 
            log1_coef, 
            log_coef, 
            quadratic_coef,
            exp_coef,
            quadratic_coef_sign,
            exp_coef_sign
        )
        np.random.seed(seed[16])
        clean_trend = np.random.choice(a=[False, True], p=[0.98, 1-0.98])
        season_train = series[start_train:end_train]
        season_target = series[end_train: (end_train + self.max_len_target)]
        trend_train = trend[start_train:end_train]
        trend_target = trend[end_train: (end_train + self.max_len_target)]
        if clean_trend:
            series = trend
        else:
            series = series + trend
        train = series[start_train:end_train]
        target = series[end_train: (end_train + self.max_len_target)]
        train_min = np.min(train)
        train_ptp = np.ptp(train)

        if trend_name == None:
            trend_min = 0
            trend_ptp = 1
            season_min = 0
            season_ptp = 1

        else:   
            trend_min = np.min(trend_train)
            trend_ptp = np.ptp(trend_train)
            season_min = np.min(season_train)
            season_ptp = np.ptp(season_train)

            season_train = (season_train - season_min) / season_ptp
            season_target = (season_target - season_min) / season_ptp
            trend_train = (trend_train - trend_min) / trend_ptp
            trend_target = (trend_target - trend_min) / trend_ptp 

        # w/o trend
        # train = (train - train_min) / train_ptp
        # target = (target - train_min) / train_ptp

        # with trend
        if clean_trend == True and trend_name == None:
            pass
        else:    
            train = (train - train_min) / train_ptp
            target = (target - train_min) / train_ptp
            
        if np.argmax(target > 2.) != 0:
            id = np.argmax(target > 2.)
            target = target[:id]
            season_target = season_target[:id]
            trend_target = trend_target[:id]
        if np.argmax(target < -1.) != 0:
            id = np.argmax(target < -1.)
            target = target[:id]
            season_target = season_target[:id]
            trend_target = trend_target[:id]
            ## add this about trend
            
        mask_target = np.pad(np.ones(len(target)), (0, self.max_len_target - len(target)), mode = 'constant')
        target = np.pad(target, (0, self.max_len_target - len(target)), mode = 'constant')
        trend_target = np.pad(trend_target, (0, self.max_len_target - len(trend_target)), mode = 'constant')
        season_target = np.pad(season_target, (0, self.max_len_target - len(season_target)), mode = 'constant')
        mask_train = np.pad(np.ones(len(train)), (0, self.max_len_train - len(train)), mode = 'constant')
        train = np.pad(train, (0, self.max_len_train - len(train)), mode = 'constant')
        # mask_signal = np.pad(np.ones(len(series)), (0, self.all_length - len(target) - len(train)), mode = 'constant')
        # signal = np.pad(signal, (0, self.all_length - len(train) - len(target)), mode = 'constant')
        mask_fft_coefs = np.pad(np.ones(len(coefs)), (0, n_c), mode = 'constant')
        coefs = np.pad(coefs, (0, n_c), mode = 'constant')
        
        if trend_name is not None:
            if trend_name == 'linear':
                trend_sign = np.sign(linear_coef)
            elif trend_name == 'log':
                trend_sign = np.sign(log_coef)
            elif trend_name == 'log1':
                trend_sign = np.sign(log1_coef)
            elif trend_name == 'exp':
                trend_sign = np.sign(exp_coef_sign)
            elif trend_name == 'quadratic':
                trend_sign = np.sign(quadratic_coef_sign)
            else:
                print('?????')
        else:
            trend_sign = 0

        if trend_name == None:
            trend_bool = False
        else:
            trend_bool = True
            
        
        out = {
            # 'series_': series.astype('float32'),
            'series': train.astype('float32'), # (train.astype('float32') - 0.5) * 2,
            'target': target.astype('float32'), # (target.astype('float32') - 0.5) * 2,
            'mask_series': mask_train,
            'mask_target': mask_target,
            'trend_sign': trend_sign,
            'trend_bool': trend_bool,
            'trend': trend_target,
            'season': season_target,
            'season_bool': not clean_trend,
            'trend_min': trend_min,
            'trend_ptp': trend_ptp,
            # 'mask_trend': mask_trend,
            # 'trend': trend,
            # 'mask_noise': mask_noise,
            # 'noise': noise,
            # # 'mask_signal': mask_signal,
            # # 'signal': series,
            # 'mask_fft_coefs': mask_fft_coefs,
            # 'fft_coefs': coefs,
            'seed': seed,
        }
        return out