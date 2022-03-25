#!/bin/bash

#for heu in {9,15,21,27}
for model in {3,}
do
  for W in {6..12}
  do
    for i in {0..4}
    do
      log_filename=random_pos_$i\_$W\_$model
      echo $log_filename
      python3 main.py --gpunum 2 --model $model --data_div 5 --val_data_num $i --batch 256 --lr 0.00002 --epochs 30 --W $W --log $log_filename --save-model
    done
  done
done
