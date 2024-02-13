export CUDA_VISIBLE_DEVICES=1
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/few-shot" ]; then
    mkdir ./logs/few-shot
fi

model_name=GPT4TS

root_path_name=../../datasets/ili/
data_path_name=ili.csv
model_id_name=ili
data_name=custom

seq_len=104
pred_len=6
gpt_layer=6

for train_budget in 52 104 156 208 260 312 364 416
do
    python -u main.py \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --data $data_name \
      --train_budget $train_budget \
      --drop_last_test 0 \
      --seq_len $seq_len \
      --label_len 18 \
      --pred_len $pred_len \
      --batch_size 16 \
      --learning_rate 0.0001 \
      --train_epochs 10 \
      --decay_fac 0.75 \
      --d_model 768 \
      --n_heads 4 \
      --d_ff 768 \
      --dropout 0.3 \
      --enc_in 7 \
      --c_out 7 \
      --freq 0 \
      --patch_size 24 \
      --stride 2 \
      --percent 100 \
      --gpt_layer $gpt_layer \
      --itr 1 \
      --model $model_name \
      --is_gpt 1 >logs/few-shot/$model_name'_'$model_id_name'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$train_budget.log 
done