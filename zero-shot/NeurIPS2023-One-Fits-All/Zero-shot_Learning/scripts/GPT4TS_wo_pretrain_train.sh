export CUDA_VISIBLE_DEVICES=0
model=GPT4TS_M4_Monthly
python train.py \
    --model $model \
    --config_path ./configs/GPT4TS_wo_pretrain/ \
    --checkpoints ./checkpoints/GPT4TS_wo_pretrain/ > ./logs/GPT4TS_wo_pretrain/$model.txt