export CUDA_VISIBLE_DEVICES=0
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/few-shot" ]; then
    mkdir ./logs/few-shot
fi

model_name=DLinear

root_path_name=../../datasets/electricity/
data_path_name=electricity.csv
model_id_name=electricity
data_name=custom

seq_len=336
pred_len=96
period=24

for j in 48 96 192 384 768 1536 3072 6144
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
      --enc_in 321 \
      --des 'Exp' \
      --itr 1 --batch_size 16 --learning_rate 0.001 >logs/few-shot/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$train_budget.log 
done