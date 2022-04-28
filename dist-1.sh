#!/usr/bin/env bash

GPUS=$1
PORT=${PORT:-29508}

for l in 0.00001 0.000005 0.000001 0.0000005 0.0000001 0.00000005 0.00000001; do
s=5;

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
/home/sjtu/scratch/zltan/mmclassification/tools/my_tent_train.py \
/home/sjtu/scratch/zltan/mim/configs/vit-cls.py --launcher pytorch --lr $l --corruption fog --severity $s;
done

