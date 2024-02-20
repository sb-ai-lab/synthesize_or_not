# ForecastPFN

This is the code repository for the paper [_ForecastPFN: Synthetically-Trained Zero-Shot Forecasting_](https://arxiv.org/abs/2311.01933). 

<!-- The codebase has these parts: 
- `./src/` contains all code to replicate the ForecastPFN synthetic data generation and training procedure
- `./benchmark/` contains all the code to replicate the benchmark of ForecastPFN against the the other baselines. 

# Table of contents
1. [Installation](#installation-)
2. [Inference with pretrained model](#inference-with-pretrained-model-)
3. [Synthetic Data Generation](#synthetic-data-generation-)
4. [Model Training](#model-training-) -->

## Get Start

- Install environment via ```conda env create -f fpfn.yml```. This repository uses Python 3.9, CUDA 11.2 and cuDNN 8.1.
- Download data. You can obtain all the benchmarks from [[TimesNet](https://github.com/thuml/Time-Series-Library)]. Make sure to put it in the folder `./academic_data/`.
- Finally, the ForecastPFN model weights should be downloaded [here](https://drive.google.com/file/d/1acp5thS7I4g_6Gw40wNFGnU1Sx14z0cU/view?usp=sharing). Make sure to put it in the folder `./saved_weights/`.
- To reproduce zero-shot resuls for benchmarks, use ```run_benchmarks.py```. Please specify data location in ```PATH```.
- To reproduce zero-shot results M4 dataset, use ```run_M4.py```. Please specify data location in ```PATH```.