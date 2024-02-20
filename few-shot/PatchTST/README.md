# A Time Series is Worth 64 Words: Long-term Forecasting with Transformers - official implementation.

Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam, "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers", ICLR, 2023. [[paper](https://arxiv.org/abs/2211.14730)]


## Get Start

- Install environment from yml file ```conda env create -f conda_baseline.yml```.
- Download data. You can obtain all the benchmarks from [[TimesNet](https://github.com/thuml/Time-Series-Library)].
- You can find scripts to reproduce results for few-shot learning in ```scripts/few-shot``` for different horizons.
- You can find scripts to reproduce results for few-shot learning in ```scripts/few-shot-horizon-6``` for different train budgets (Table 15 in Appendix).

## Citation

```
@inproceedings{nie2022time,
  title={A Time Series is Worth 64 Words: Long-term Forecasting with Transformers},
  author={Nie, Yuqi and Nguyen, Nam H and Sinthong, Phanwadee and Kalagnanam, Jayant},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2022}
}
```