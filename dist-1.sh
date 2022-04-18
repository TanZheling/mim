#!/usr/bin/env bash

GPUS=$1
PORT=${PORT:-29508}

for l in 0.1 0.05; do
s=5;

CUDA_VISIBLE_DEVICES=0,2,3 PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
/home/sjtu/scratch/zltan/mmclassification/tools/my_tent_train.py \
/home/sjtu/scratch/zltan/mim/configs/vit-contrast.py --launcher pytorch --lr $l --corruption fog --severity $s;
done

