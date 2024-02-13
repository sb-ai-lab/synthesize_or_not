export CUDA_VISIBLE_DEVICES=0
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/few-shot" ]; then
    mkdir ./logs/few-shot
fi

model_name=DLinear

root_path_name=../../datasets/ETT/
data_path_name=traffic.csv
model_id_name=traffic
data_name=traffic

seq_len=336
pred_len=96
period=168

for j in 336 672 1344 2688 5376 10752 21504 43008
do
    train_budget=$(($j))
    python -u run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --train_budget $train_budget \
      --drop_last_test 0 \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 862 \
      --des 'Exp' \
      --itr 1 --batch_size 16 --learning_rate 0.05 >logs/few-shot/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$train_budget.log 
done