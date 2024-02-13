export CUDA_VISIBLE_DEVICES=2
for model in \
GPT4TS_M3_Yearly GPT4TS_M3_Quarterly GPT4TS_M3_Monthly GPT4TS_M3_Other \
GPT4TS_M4_Yearly GPT4TS_M4_Quarterly GPT4TS_M4_Monthly GPT4TS_M4_Weekly GPT4TS_M4_Daily GPT4TS_M4_Hourly \
DLinear_M3_Yearly DLinear_M3_Quarterly DLinear_M3_Monthly DLinear_M3_Other \
DLinear_M4_Yearly DLinear_M4_Quarterly DLinear_M4_Monthly DLinear_M4_Weekly DLinear_M4_Daily DLinear_M4_Hourly \
PatchTST_M3_Yearly PatchTST_M3_Quarterly PatchTST_M3_Monthly PatchTST_M3_Other \
PatchTST_M4_Yearly PatchTST_M4_Quarterly PatchTST_M4_Monthly PatchTST_M4_Weekly PatchTST_M4_Daily PatchTST_M4_Hourly
do
for target_data in Weather ECL ILI Traffic ETT ETTh1 ETTh2 ETTm1 ETTm2
do
for source_scaler in standard_scaler min_max_scaler quantile_transformer False
do
python inference.py \
    --model $model \
    --target_data $target_data \
    --checkpoints ./checkpoints/ \
    --res_path ./results/scale_check_on_val/$model'_'$target_data'_'$source_scaler'.csv' \
    --source_scaling $source_scaler >> ./logs/scaling_check/$model.txt \
    --test_on_val 0
done
done
done

