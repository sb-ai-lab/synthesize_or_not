# Our synthetic data

Here, you can read how to reproduce paper results for our model and synthetic series.

## Get Start

- Install environment via ```conda env create -f time_series.yml```.
- To reproduce few-shot resuls for benchmarks, use ```run_benchmarks_few_shot.py```. Please specify checkpoint and data location in ```CH_PATH``` and ```PATH```.
- To reproduce zero-shot results for benchmarks and M4 dataset, use ```run_benchmarks_zero_shot.py``` and ```run_M.py```. Please specify checkpoint and data location in ```CH_PATH``` and ```PATH```.