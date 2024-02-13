for model in \
GPT4TS_M3_Yearly GPT4TS_M3_Quarterly GPT4TS_M3_Monthly GPT4TS_M3_Other \
GPT4TS_M4_Yearly GPT4TS_M4_Quarterly GPT4TS_M4_Monthly GPT4TS_M4_Weekly GPT4TS_M4_Daily GPT4TS_M4_Hourly \
DLinear_M3_Yearly DLinear_M3_Quarterly DLinear_M3_Monthly DLinear_M3_Other \
DLinear_M4_Yearly DLinear_M4_Quarterly DLinear_M4_Monthly DLinear_M4_Weekly DLinear_M4_Daily DLinear_M4_Hourly \
PatchTST_M3_Yearly PatchTST_M3_Quarterly PatchTST_M3_Monthly PatchTST_M3_Other \
PatchTST_M4_Yearly PatchTST_M4_Quarterly PatchTST_M4_Monthly PatchTST_M4_Weekly PatchTST_M4_Daily PatchTST_M4_Hourly
do
python train.py \
    --model $model \
    --config_path ./configs/ \
    --checkpoints ./checkpoints/ 
done