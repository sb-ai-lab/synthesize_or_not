export CUDA_VISIBLE_DEVICES=4
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/few-shot" ]; then
    mkdir ./logs/few-shot
fi

model_name=iTransformer

root_path_name=../datasets/ETT/
data_path_name=ETTm2.csv
model_id_name=ettm2
data_name=ETTm2

seq_len=96
pred_len=96
period=48

for j in 96 192 384 768 1536 3072 6144 12288
do
    train_budget=$(($j))
    python -u run.py \
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
      --e_layers 2 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --d_model 128 \
      --d_ff 128 \
      --itr 1 >logs/few-shot/$model_name'_'$model_id_name'_0_'$seq_len'_'$pred_len'_'$train_budget.log 
done