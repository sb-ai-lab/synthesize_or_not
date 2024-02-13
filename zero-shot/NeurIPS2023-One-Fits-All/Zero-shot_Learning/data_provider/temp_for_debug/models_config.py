from datasets_config import *

class ModelParams:
  # general params
  n_heads=16
  e_layers=3
  d_layers=3
  d_ff=512
  enc_in=862
  dec_in=862
  c_out=862
  patch_size=16
  stride=8
  pretrain=1
  freeze=1
  max_len=-1
  tmax=10
  cos=0
  train_all=0
  look_bask_num=12
  proj_hid=8
  top_k=5
  num_kernels=6
  activation='gelu'
  output_attention='store_true'
  factor=1
  p_hidden_dims=[128, 128]
  p_hidden_layers=2
  moving_avg=25
  distill=True
  modes=32
  train_all=1

  # train params
  loss_func='mse'
  percent=100
  train_epochs=10
  lradj='type1'
  decay_fac=0.75
  kernel_size=25
  print_int=1000
  dropout=0.2
  
  # inference params
  test_batch_size=128
  test_label_len=48


# Params for GPT4TS are taken from https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All/tree/main/Zero-shot_Learning
class GPT4TSParams(ModelParams):
  # general params
  model='GPT4TS'
  #seq_len=104
  seq_len=6
  d_model=768
  stride=1
  patch_size=1
  gpt_layers=6
  is_gpt=1

  # train params
  train_epochs=10
  dropout=0


class GPT4TS_M3_YearlyParams(GPT4TSParams):
  source_dataset_params = M3_YearlyParams
  # general params
  label_len=0

  # train params  
  batch_size=32
  learning_rate=0.002
  decay_fac=0.5


class GPT4TS_M3_QuarterlyParams(GPT4TSParams):
  source_dataset_params = M3_QuarterlyParams
  # general params
  label_len=0
  patch_size=2
  proj_hid=64

  # train params  
  batch_size=64
  learning_rate=0.002
  decay_fac=1


class GPT4TS_M3_MonthlyParams(GPT4TSParams):
  source_dataset_params = M3_MonthlyParams
  # general params
  label_len=10

  # train params
  batch_size=32 
  learning_rate=0.0001
  decay_fac=0.5


class GPT4TS_M3_OtherParams(GPT4TSParams):
  source_dataset_params = M3_OtherParams
  # general params
  label_len=10

  # train params
  batch_size=16 
  learning_rate=0.01
  decay_fac=0.75


class GPT4TS_M4_YearlyParams(GPT4TSParams):
  source_dataset_params = M4_YearlyParams
  # general params
  label_len=0

  # train params  
  train_epochs=20
  batch_size=512
  learning_rate=0.001
  decay_fac=0.75
  
  
class GPT4TS_M4_QuarterlyParams(GPT4TSParams):
  source_dataset_params = M4_QuarterlyParams
  # general params
  label_len=0

  # train params  
  batch_size=512
  learning_rate=0.01
  decay_fac=0.75
  

class GPT4TS_M4_MonthlyParams(GPT4TSParams):
  source_dataset_params = M4_MonthlyParams
  # general params
  label_len=0
  patch_size=2
  stride=2

  # train params  
  batch_size=2048
  learning_rate=0.001
  decay_fac=0.75
  

class GPT4TS_M4_WeeklyParams(GPT4TSParams):
  source_dataset_params = M4_WeeklyParams
  # not provided, made similar to M4_Monthly
  # general params
  label_len=0
  patch_size=2
  stride=2

  # train params  
  batch_size=2048
  learning_rate=0.001
  decay_fac=0.75


class GPT4TS_M4_DailyParams(GPT4TSParams):
  source_dataset_params = M4_DailyParams
  # not provided, made similar to M4_Hourly
  # general params
  label_len=0
  patch_size=2

  # train params  
  batch_size=512
  learning_rate=0.005
  decay_fac=0.5


class GPT4TS_M4_HourlyParams(GPT4TSParams):
  source_dataset_params = M4_HourlyParams
  # general params
  label_len=0
  patch_size=2

  # train params  
  batch_size=512
  learning_rate=0.005
  decay_fac=0.5

# Params for DLinear are taken from https://github.com/thuml/Time-Series-Library/blob/main/scripts/short_term_forecast/DLinear_M4.sh
# Assume that params for M3 can be derived from params for M4
# And architecture of model is the same for all granularities of M4
class DLinearParams(ModelParams):
  # general params
  model='DLinear'
  seq_len=104
  label_len=48
  d_model=512
  e_layers=2
  d_layers=1
  factor=3
  enc_in=1
  dec_in=1
  c_out=1
  is_gpt=0
  
  # train params
  batch_size=16
  learning_rate=0.001
  
  
class DLinear_M3_YearlyParams(DLinearParams):
  source_dataset_params = M3_YearlyParams


class DLinear_M3_QuarterlyParams(DLinearParams):
  source_dataset_params = M3_QuarterlyParams


class DLinear_M3_MonthlyParams(DLinearParams):
  source_dataset_params = M3_MonthlyParams


class DLinear_M3_OtherParams(DLinearParams):
  source_dataset_params = M3_OtherParams


class DLinear_M4_YearlyParams(DLinearParams):
  source_dataset_params = M4_YearlyParams
  
  
class DLinear_M4_QuarterlyParams(DLinearParams):
  source_dataset_params = M4_QuarterlyParams
  

class DLinear_M4_MonthlyParams(DLinearParams):
  source_dataset_params = M4_MonthlyParams
  

class DLinear_M4_WeeklyParams(DLinearParams):
  source_dataset_params = M4_WeeklyParams


class DLinear_M4_DailyParams(DLinearParams):
  source_dataset_params = M4_DailyParams


class DLinear_M4_HourlyParams(DLinearParams):
  source_dataset_params = M4_HourlyParams
  

# Params for PatchTST are taken from https://github.com/thuml/Time-Series-Library/blob/main/scripts/long_term_forecast/ILI_script/PatchTST.sh
# Assume that params for M3 can be derived from params for M4
# And architecture of model is the same for all granularities of M4
class PatchTSTParams(ModelParams):
  # general params
  model='PatchTST'
  seq_len=148
  label_len=18
  n_heads=16
  d_model=2048
  e_layers=4
  d_layers=1
  factor=3
  enc_in=1
  dec_in=1
  c_out=1
  is_gpt=0
  
  # train params
  batch_size=16
  learning_rate=0.001


class PatchTST_M3_YearlyParams(PatchTSTParams):
  source_dataset_params = M3_YearlyParams


class PatchTST_M3_QuarterlyParams(PatchTSTParams):
  source_dataset_params = M3_QuarterlyParams


class PatchTST_M3_MonthlyParams(PatchTSTParams):
  source_dataset_params = M3_MonthlyParams


class PatchTST_M3_OtherParams(PatchTSTParams):
  source_dataset_params = M3_OtherParams


class PatchTST_M4_YearlyParams(PatchTSTParams):
  source_dataset_params = M4_YearlyParams
  
  
class PatchTST_M4_QuarterlyParams(PatchTSTParams):
  source_dataset_params = M4_QuarterlyParams
  

class PatchTST_M4_MonthlyParams(PatchTSTParams):
  source_dataset_params = M4_MonthlyParams
  

class PatchTST_M4_WeeklyParams(PatchTSTParams):
  source_dataset_params = M4_WeeklyParams


class PatchTST_M4_DailyParams(PatchTSTParams):
  source_dataset_params = M4_DailyParams


class PatchTST_M4_HourlyParams(PatchTSTParams):
  source_dataset_params = M4_HourlyParams
  
  
# Factory Object
class ModelsFactory:
    def __init__(self):
        self.models = {
            "GPT4TS_M3_Yearly": GPT4TS_M3_YearlyParams,
            "GPT4TS_M3_Quarterly": GPT4TS_M3_QuarterlyParams,
            "GPT4TS_M3_Monthly": GPT4TS_M3_MonthlyParams,
            "GPT4TS_M3_Other": GPT4TS_M3_OtherParams,
            "GPT4TS_M4_Yearly": GPT4TS_M4_YearlyParams,
            "GPT4TS_M4_Quarterly": GPT4TS_M4_QuarterlyParams,
            "GPT4TS_M4_Monthly": GPT4TS_M4_MonthlyParams,
            "GPT4TS_M4_Weekly": GPT4TS_M4_WeeklyParams,
            "GPT4TS_M4_Daily": GPT4TS_M4_DailyParams,
            "GPT4TS_M4_Hourly": GPT4TS_M4_HourlyParams,

            "DLinear_M3_Yearly": DLinear_M3_YearlyParams,
            "DLinear_M3_Quarterly": DLinear_M3_QuarterlyParams,
            "DLinear_M3_Monthly": DLinear_M3_MonthlyParams,
            "DLinear_M3_Other": DLinear_M3_OtherParams,
            "DLinear_M4_Yearly": DLinear_M4_YearlyParams,
            "DLinear_M4_Quarterly": DLinear_M4_QuarterlyParams,
            "DLinear_M4_Monthly": DLinear_M4_MonthlyParams,
            "DLinear_M4_Weekly": DLinear_M4_WeeklyParams,
            "DLinear_M4_Daily": DLinear_M4_DailyParams,
            "DLinear_M4_Hourly": DLinear_M4_HourlyParams,

            "PatchTST_M3_Yearly": PatchTST_M3_YearlyParams,
            "PatchTST_M3_Quarterly": PatchTST_M3_QuarterlyParams,
            "PatchTST_M3_Monthly": PatchTST_M3_MonthlyParams,
            "PatchTST_M3_Other": PatchTST_M3_OtherParams,
            "PatchTST_M4_Yearly": PatchTST_M4_YearlyParams,
            "PatchTST_M4_Quarterly": PatchTST_M4_QuarterlyParams,
            "PatchTST_M4_Monthly": PatchTST_M4_MonthlyParams,
            "PatchTST_M4_Weekly": PatchTST_M4_WeeklyParams,
            "PatchTST_M4_Daily": PatchTST_M4_DailyParams,
            "PatchTST_M4_Hourly": PatchTST_M4_HourlyParams,
        }

    def get_allowed(self):
        return sorted(list(self.models.keys()))

    def __getitem__(self, model_name):
        return self.models[model_name]