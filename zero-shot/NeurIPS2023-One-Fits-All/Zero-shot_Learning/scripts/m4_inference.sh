export CUDA_VISIBLE_DEVICES=0
for model in \
DLinear_M3_Yearly DLinear_M3_Quarterly DLinear_M3_Monthly DLinear_M3_Other \
PatchTST_M3_Yearly PatchTST_M3_Quarterly PatchTST_M3_Monthly PatchTST_M3_Other \
GPT4TS_M3_Yearly GPT4TS_M3_Quarterly GPT4TS_M3_Monthly GPT4TS_M3_Other
do
for target_data in M4_Yearly M4_Quarterly M4_Monthly M4_Weekly M4_Daily M4_Hourly
do
python inference.py \
    --model $model \
    --target_data $target_data \
    --checkpoints ./checkpoints/ \
    --test_on_val 0 \
    --res_path ./results/M4_test/$model'_'$target_data'.csv' \
    --source_scaling False >> ./logs/M4_test/$model.txt
done
done
