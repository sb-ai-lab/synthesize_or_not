export CUDA_VISIBLE_DEVICES=2
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/few-shot-horizon-6" ]; then
    mkdir ./logs/few-shot-horizon-6
fi

model_name=DLinear

root_path_name=../../datasets/illness/
data_path_name=national_illness.csv
model_id_name=national_illness
data_name=custom

seq_len=104
pred_len=6

for train_budget in 52 104 156 208 260 312 364 416
do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --train_budget $train_budget \
      --drop_last_test 0 \
      --seq_len $seq_len \
      --label_len 18 \
      --pred_len $pred_len \
      --enc_in 7 \
      --des 'Exp' \
      --itr 1 --batch_size 32 --learning_rate 0.01 >logs/few-shot-horizon-6/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$train_budget.log 
done