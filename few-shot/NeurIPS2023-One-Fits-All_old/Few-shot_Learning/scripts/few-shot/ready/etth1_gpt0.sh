export CUDA_VISIBLE_DEVICES=2
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/few-shot" ]; then
    mkdir ./logs/few-shot
fi

model_name=GPT4TS

root_path_name=../../datasets/ETT/
data_path_name=ETTh1.csv
model_id_name=etth1
data_name=ett_h

seq_len=336
pred_len=96
period=12

for j in 24 48 96 192 384 768 1536 3072
do
    train_budget=$(($j))
    python -u main.py \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --data $data_name \
      --train_budget $train_budget \
      --drop_last_test 0 \
      --seq_len $seq_len \
      --label_len 168 \
      --pred_len $pred_len \
      --batch_size 256 \
      --lradj type4 \
      --decay_fac 0.5 \
      --learning_rate 0.001 \
      --train_epochs 10 \
      --d_model 768 \
      --n_heads 4 \
      --d_ff 768 \
      --dropout 0.3 \
      --enc_in 7 \
      --c_out 7 \
      --freq 0 \
      --patch_size 16 \
      --stride 8 \
      --percent 100 \
      --gpt_layer 0 \
      --itr 1 \
      --model $model_name \
      --cos 1 \
      --tmax 20 \
      --is_gpt 1 >logs/few-shot/$model_name'_'$model_id_name'_0_'$seq_len'_'$pred_len'_'$train_budget.log 
done