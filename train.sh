#!/bin/bash

if [ $# -ge 3 ];then
  CONFIG_FILE=$3
else
  CONFIG_FILE='training_config.ini'
fi

#for heu in {9,15,21,27}

epochs=$(awk '/^EPOCHS/{print $3}' ${CONFIG_FILE})
patience=$(awk '/^PATIENCE/{print $3}' ${CONFIG_FILE})
model=$(awk '/^MODEL/{print $3}' ${CONFIG_FILE})
aug_ratio=$(awk '/^AUG_RATIO/{print $3}' ${CONFIG_FILE})
batch=$(awk '/^BATCH/{print $3}' ${CONFIG_FILE})
lr=$(awk '/^LEARNING_RATE/{print $3}' ${CONFIG_FILE})
noise=$(awk '/^NOISE_ADDED/{print $3}' ${CONFIG_FILE})
log_prefix=$(awk '/^LOG_PREFIX/{print $3}' ${CONFIG_FILE})
dataset_postfix=$(awk '/^DATA_POSTFIX/{print $3}' ${CONFIG_FILE})

gpunum=$1


for W in {6..12}
do
  for i in {0..4}
  do
#      if [ $# -ge 6 ];then
#        W_start=$5
#        V_start=$6
#        if [ $W -lt $W_start ];then
#          echo $W\_$i pass
#          continue
#        elif [ $W -eq $W_start -a $i -lt $V_start ];then
#          echo $W\_$i pass
#          continue
#        fi
#      fi
#
    if [ $# -ge 2 ];then
      V_specific=$2
      if [ $i -ne $V_specific ];then
        echo $W\_$i pass
        continue
      fi
    fi
    log_filename=$log_prefix\_$noise\_$model\_$lr\_$aug_ratio\_$W\_$i
    echo $log_filename
    export CUDA_CACHE_DISABLE=1
    export LD_PRELOAD=/usr/local/lib/libjemalloc.so
    export LRU_CACHE_CAPACITY=1
    python3 training.py --gpunum $gpunum --seed $i --batch-size $batch --lr $lr --dataset-postfix $dataset_postfix --epochs $epochs --patience $patience --W $W --log $log_filename --model $model --aug-ratio $aug_ratio --test-batch-size 4096 --save-model #--batch-multiplier 8 --dry-run
  done
done

