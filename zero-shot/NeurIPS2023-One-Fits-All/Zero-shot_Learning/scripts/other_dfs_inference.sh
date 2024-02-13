export CUDA_VISIBLE_DEVICES=1
for model in \
GPT4TS_M3_Quarterly GPT4TS_M3_Monthly GPT4TS_M3_Other \
GPT4TS_M4_Yearly GPT4TS_M4_Quarterly GPT4TS_M4_Monthly GPT4TS_M4_Weekly GPT4TS_M4_Daily GPT4TS_M4_Hourly \
DLinear_M4_Yearly DLinear_M4_Quarterly DLinear_M4_Monthly DLinear_M4_Weekly DLinear_M4_Daily DLinear_M4_Hourly \
PatchTST_M3_Yearly PatchTST_M3_Quarterly PatchTST_M3_Monthly PatchTST_M3_Other \
PatchTST_M4_Yearly PatchTST_M4_Quarterly PatchTST_M4_Monthly PatchTST_M4_Weekly PatchTST_M4_Daily PatchTST_M4_Hourly
do
for target_data in Weather ECL ILI Traffic ETTh1 ETTh2 ETTm1 ETTm2
do
python inference.py \
    --model $model \
    --target_data $target_data \
    --checkpoints ./checkpoints/ \
    --test_on_val 0 \
    --res_path ./results/other_dfs_inference/$model'_'$target_data'.csv' \
    --source_scaling standard_scaler >> ./logs/other_dfs_inference/$model.txt
done
done

for model in \
DLinear_M3_Yearly DLinear_M3_Quarterly DLinear_M3_Monthly DLinear_M3_Other
do
for target_data in Weather ECL ILI Traffic ETT ETTh1 ETTh2 ETTm1 ETTm2
do
python inference.py \
    --model $model \
    --target_data $target_data \
    --checkpoints ./checkpoints/ \
    --test_on_val 0 \
    --res_path ./results/other_dfs_inference/$model'_'$target_data'.csv' \
    --source_scaling False >> ./logs/other_dfs_inference/$model.txt
done
done