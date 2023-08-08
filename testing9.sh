#!/bin/bash

#for heu in {9,15,21,27}
model=3
W=9
i=0
log_filename=random_pos_$i\_$W\_$model
echo $log_filename
python3 main.py --gpunum 1 --model $model --data_div 5 --val_data_num $i --batch 256 --lr 0.00002 --epochs 30 --W $W --dry-run --log $log_filename --test random_pos_0_9_3 #--save-model
