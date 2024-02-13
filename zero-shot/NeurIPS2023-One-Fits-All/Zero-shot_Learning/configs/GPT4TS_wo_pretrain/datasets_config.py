import pathlib

ROOT_DATASET_PATH = pathlib.Path(__file__).parent.parent.parent / 'data'


class DatasetParams:
  horizon=None
  dataset_type=None
  path=None
  features='M'
  freq=None   # freq=1
  season=None
  target='OT'
  embed=None  # embed='timeF'
  num_workers=10


class TransformersBenchmarkParams(DatasetParams):
  #target_horizon=96
  target_horizon=6
  dataset_type='custom'


class WeatherParams(TransformersBenchmarkParams):
  root_path = str(ROOT_DATASET_PATH / 'weather')
  #target_horizon=24
  data_path = ('weather.csv')


class ECLParams(TransformersBenchmarkParams):
  root_path = str(ROOT_DATASET_PATH / 'ecl')
  data_path = ('ecl.csv')


class ILIParams(TransformersBenchmarkParams):
  root_path = str(ROOT_DATASET_PATH / 'ili')
  data_path = ('ili.csv')


class TrafficParams(TransformersBenchmarkParams):
  root_path = str(ROOT_DATASET_PATH / 'traffic')
  data_path = ('traffic.csv')


class ETThParams(TransformersBenchmarkParams):
  dataset_type='ett_h'
  
  
class ETTmParams(TransformersBenchmarkParams):
  dataset_type='ett_m'


class ETTh1Params(ETThParams):
  root_path = str(ROOT_DATASET_PATH / 'ett')
  data_path = ('etth1.csv')


class ETTh2Params(ETThParams):
  root_path = str(ROOT_DATASET_PATH / 'ett')
  data_path = ('etth2.csv')


class ETTm1Params(ETTmParams):
  root_path = str(ROOT_DATASET_PATH / 'ett')
  data_path = ('ettm1.csv')


class ETTm2Params(ETTmParams):
  root_path = str(ROOT_DATASET_PATH / 'ett')
  data_path = ('ettm2.csv')


class MBenchmarkParams(DatasetParams):
  target_horizon=6
  dataset_type='tsf_data'


class M4_YearlyParams(MBenchmarkParams):
  horizon=6
  root_path = str(ROOT_DATASET_PATH / 'm4')
  data_path = ('m4_yearly.tsf')


class M4_QuarterlyParams(MBenchmarkParams):
  horizon=8
  root_path = str(ROOT_DATASET_PATH / 'm4')
  data_path = ('m4_quarterly.tsf')


class M4_MonthlyParams(MBenchmarkParams):
  horizon=18
  root_path = str(ROOT_DATASET_PATH / 'm4')
  data_path = ('m4_monthly.tsf')


class M4_WeeklyParams(MBenchmarkParams):
  horizon=13
  root_path = str(ROOT_DATASET_PATH / 'm4')
  data_path = ('m4_weekly.tsf')


class M4_DailyParams(MBenchmarkParams):
  horizon=14
  root_path = str(ROOT_DATASET_PATH / 'm4')
  data_path = ('m4_daily.tsf')


class M4_HourlyParams(MBenchmarkParams):
  horizon=48
  root_path = str(ROOT_DATASET_PATH / 'm4')
  data_path = ('m4_hourly.tsf')


class M3_YearlyParams(MBenchmarkParams):
  horizon=6
  root_path = str(ROOT_DATASET_PATH / 'm3')
  data_path = ('m3_yearly.tsf')


class M3_QuarterlyParams(MBenchmarkParams):
  horizon=8
  root_path = str(ROOT_DATASET_PATH / 'm3')
  data_path = ('m3_quarterly.tsf')


class M3_MonthlyParams(MBenchmarkParams):
  horizon=18
  root_path = str(ROOT_DATASET_PATH / 'm3')
  data_path = ('m3_monthly.tsf')


class M3_OtherParams(MBenchmarkParams):
  horizon=8
  root_path = str(ROOT_DATASET_PATH / 'm3')
  data_path = ('m3_other.tsf')


# Factory Object
class DatasetsFactory:
    def __init__(self):
        self.datasets = {
            "Weather": WeatherParams,
            "ECL": ECLParams,
            "ILI": ILIParams,
            "Traffic": TrafficParams,
            "ETTh1": ETTh1Params,
            "ETTh2": ETTh2Params,
            "ETTm1": ETTm1Params,
            "ETTm2": ETTm2Params,
            
            "M4_Yearly": M4_YearlyParams,
            "M4_Quarterly": M4_QuarterlyParams,
            "M4_Monthly": M4_MonthlyParams,
            "M4_Weekly": M4_WeeklyParams,
            "M4_Daily": M4_DailyParams,
            "M4_Hourly": M4_HourlyParams,
            
            "M3_Yearly": M3_YearlyParams,
            "M3_Quarterly": M3_QuarterlyParams,
            "M3_Monthly": M3_MonthlyParams,
            "M3_Other": M3_OtherParams,
        }

    def get_allowed(self):
        return sorted(list(self.datasets.keys()))

    def __getitem__(self, dataset_name):
        return self.datasets[dataset_name]