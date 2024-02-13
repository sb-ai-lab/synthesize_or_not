
for patch in 16
do
for stride in 8
do
for lr in 0.0005
do

python src/main.py \
    --output_dir experiments \
    --comment "classification from Scratch" \
    --name PEMS-SF \
    --records_file Classification_records.xls \
    --data_dir ./datasets/PEMS-SF \
    --data_class tsra \
    --pattern TRAIN \
    --val_pattern TEST \
    --epochs 50 \
    --lr $lr \
    --patch_size $patch \
    --stride $stride \
    --optimizer RAdam \
    --d_model 768 \
    --pos_encoding learnable \
    --task classification \
    --key_metric accuracy

done
done
done