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
  pretrain=0
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
  decay_fac=0.75
  lradj='type1'
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
  seq_len=104
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
  # not provided, made similar to traffic and ettm2 long-term 
  # due to the fact that with params for M4_Weekly need a lot of time to train
  # general params
  label_len=0
  patch_size=16
  stride=16

  # train params  
  batch_size=2048
  learning_rate=0.001
  decay_fac=0.75
  patience=3 
  cos=1


class GPT4TS_M4_HourlyParams(GPT4TSParams):
  source_dataset_params = M4_HourlyParams
  # general params
  label_len=0
  patch_size=2

  # train params  
  batch_size=512
  learning_rate=0.005
  decay_fac=0.5

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
        }

    def get_allowed(self):
        return sorted(list(self.models.keys()))

    def __getitem__(self, model_name):
        return self.models[model_name]