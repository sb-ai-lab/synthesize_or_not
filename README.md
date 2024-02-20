# Towards foundation time series model: to synthesize or not to synthesize?

Kseniia Kuvshinova, Olga Tsymboi, Alina Kostromina, Dmitry Simakov, Elizaveta Kovtun, "Towards foundation time series model: to synthesize or not to synthesize?".

The industry is rich in cases when we are required to make forecasting for large amounts of time series at once. However, we might be in a situation where we can not afford to train a separate model for each of them. Such issue in time series modeling remains without due attention. The remedy for this setting is the establishment of a foundation model. Such a model is expected to work in zero-shot and few-shot regimes. However, what should we take as a training dataset for such kind of model?

Witnessing the benefits from the enrichment of NLP datasets with artificially-generated data, we might want to adopt their experience for time series. In contrast to natural language, the process of generation of synthetic time series data is even more favorable because it provides full control of series patterns, time horizons, and number of samples. In this work, we consider the essential question if it is advantageous to train a foundation model on synthetic data or it is better to utilize only a limited number of real-life examples. Our experiments are conducted only for regular time series and speak in favor of leveraging solely the real time series. The choice of the proper authentic dataset strongly influences the performance during inference. However, in case of access even to a small amount of short time series, the switching from zero-short setting to the regime of supervised training will mostly enhance performance.

<div align="center"><img src=./pic/share_of_wins_mae.pdf width=80% /></div>

## Get Start

- Follow the instructions provided in the respective task and model folder.


## Citation

If you find this repo useful, please cite our paper. 

```
@inproceedings{zhou2023onefitsall,
  title={{One Fits All}: Power General Time Series Analysis by Pretrained LM},
  author={Tian Zhou, Peisong Niu, Xue Wang, Liang Sun, Rong Jin},
  booktitle={NeurIPS},
  year={2023}
}
```