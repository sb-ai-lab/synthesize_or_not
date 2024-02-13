model=GPT4TS

seq_len=6
pred_len=14
test_seq_len=6
test_pred_len=14

python main_test.py \
    --root_path ./data/m4/ \
    --test_root_path ./data/m3/ \
    --data_path m4_daily.tsf \
    --test_data_path m3_monthly.tsf \
    --model_id m3Darly'_'$model \
    --data tsf_data \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --test_seq_len $seq_len \
    --test_pred_len $test_pred_len \
    --label_len 0 \
    --batch_size 512 \
    --test_batch_size 128 \
    --learning_rate 0.001 \
    --train_epochs 20 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 16 \
    --d_ff 512 \
    --loss_func smape \
    --dropout 0 \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --patch_size 1 \
    --stride 1 \
    --print_int 1000 \
    --train_all 1 \
    --percent 100
