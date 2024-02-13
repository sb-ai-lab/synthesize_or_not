export CUDA_VISIBLE_DEVICES=0
model=GPT4TS_M4_Monthly
for target_data in Weather ECL ILI Traffic ETTh1 ETTh2 ETTm1 ETTm2
do
python inference.py \
    --model $model \
    --target_data $target_data \
    --checkpoints ./checkpoints/GPT4TS_wo_pretrain \
    --test_on_val 0 \
    --res_path ./results/GPT4TS_wo_pretrain/$model'_'$target_data'.csv' \
    --source_scaling standard_scaler >> ./logs/GPT4TS_wo_pretrain/$model.txt
done
