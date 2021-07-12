#!/bin/bash

#for heu in {9,15,21,27}
for model in {1,2,3,4}
do
  for heu in {27,}
  do
    for i in {1..5}
    do
      log_filename=random_pos_$i\_$heu\_$model.csv
      echo $log_filename
      python3 main.py --gpunum 3 --model $model --data_div 5 --val_data_num $i --batch 8 --lr 0.25 --epochs 15 --heu $heu --log $log_filename
    done
  done
done
