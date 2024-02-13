export CUDA_VISIBLE_DEVICES=2
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/few-shot-horizon-6" ]; then
    mkdir ./logs/few-shot-horizon-6
fi

model_name=PatchTST

root_path_name=../../datasets/traffic/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom

random_seed=2024
seq_len=148
pred_len=6

for train_budget in 336 672 1344 2688 5376 10752 21504 43008
do
    python -u run_longExp.py \
      --random_seed $random_seed \
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
      --pred_len $pred_len \
      --enc_in 862 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --lradj 'TST'\
      --itr 1 --batch_size 24 --learning_rate 0.0001 >logs/few-shot-horizon-6/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$train_budget.log 
done