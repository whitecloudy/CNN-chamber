#!/bin/bash

#for heu in {9,15,21,27}
echo $#
if [ $# -lt 4 ];then
  echo "Need more parameter"
else
  epochs=$1
  patience=$2
  model=$3
  aug_ratio=$4
  for W in {6..12}
  do
    for i in {0..4}
    do
      if [ $# -ge 6 ];then
        W_start=$5
        V_start=$6
        if [ $W -lt $W_start ];then
          echo $W\_$i pass
          continue
        elif [ $W -eq $W_start -a $i -lt $V_start ];then
          echo $W\_$i pass
          continue
        fi
      fi

      if [ $# -ge 7 ];then
        V_specific=$7
        if [ $i -ne $V_specific ];then
          echo $W\_$i pass
          continue
        fi
      fi
      log_filename=$model\_$aug_ratio\_$W\_$i
      echo $log_filename
      export CUDA_CACHE_DISABLE=1
      export LD_PRELOAD=/usr/local/lib/libjemalloc.so
      export LRU_CACHE_CAPACITY=1
      # python3 training.py --gpunum 0 --data_div 5 --val_data_num $i --batch-size 256 --lr 0.001 --epochs $epochs --patience $patience --W $W --log $log_filename --model $model --aug-ratio $aug_ratio --test-batch-size 5000 --save-model #--batch-multiplier 8 --dry-run
      python3 training.py --gpunum 1 --seed $i --batch-size 256 --lr 0.001 --epochs $epochs --patience $patience --W $W --log $log_filename --model $model --aug-ratio $aug_ratio --test-batch-size 5000 --save-model #--batch-multiplier 8 --dry-run

    done
  done
fi


