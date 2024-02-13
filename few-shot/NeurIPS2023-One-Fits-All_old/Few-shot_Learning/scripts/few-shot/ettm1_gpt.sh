export CUDA_VISIBLE_DEVICES=1
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/few-shot" ]; then
    mkdir ./logs/few-shot
fi

model_name=GPT4TS

root_path_name=../../datasets/ETT/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ett_m

seq_len=104
pred_len=6
gpt_layer=6

for train_budget in 96 192 384 768 1536 3072 6144 12288
do
    python -u main.py \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --data $data_name \
      --train_budget $train_budget \
      --drop_last_test 0 \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --batch_size 256 \
      --learning_rate 0.001 \
      --train_epochs 10 \
      --decay_fac 0.75 \
      --d_model 768 \
      --n_heads 4 \
      --d_ff 768 \
      --dropout 0.3 \
      --enc_in 7 \
      --c_out 7 \
      --freq 0 \
      --patch_size 16 \
      --stride 16 \
      --percent 100 \
      --gpt_layer $gpt_layer \
      --itr 1 \
      --model $model_name \
      --cos 1 \
      --is_gpt 1 >logs/few-shot/$model_name'_'$model_id_name'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$train_budget.log 
done