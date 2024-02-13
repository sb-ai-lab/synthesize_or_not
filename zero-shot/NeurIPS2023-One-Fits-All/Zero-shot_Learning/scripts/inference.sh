export CUDA_VISIBLE_DEVICES=0
for model in \
GPT4TS_M3_Yearly GPT4TS_M3_Quarterly GPT4TS_M3_Monthly GPT4TS_M3_Other
do
for target_data in Weather ECL ILI Traffic ETT ETTh1 ETTh2 ETTm1 ETTm2
do
python inference.py \
    --model $model \
    --target_data $target_data \
    --checkpoints ./checkpoints/ \
    --res_path ./results/$model'_'$target_data'.csv' \
    --source_scaling False >> ./logs/$model.txt
done
done

export CUDA_VISIBLE_DEVICES=0
for model in \
GPT4TS_M4_Yearly GPT4TS_M4_Quarterly GPT4TS_M4_Weekly GPT4TS_M4_Daily GPT4TS_M4_Daily GPT4TS_M4_Hourly
do
for target_data in Weather ECL ILI Traffic ETT ETTh1 ETTh2 ETTm1 ETTm2
do
python inference.py \
    --model $model \
    --target_data $target_data \
    --checkpoints ./checkpoints/ \
    --res_path ./results/$model'_'$target_data'.csv' \
    --source_scaling standard_scaler >> ./logs/$model.txt
done
done
