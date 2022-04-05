#!/usr/bin/env bash

GPUS=$1
PORT=${PORT:-29506}

for c in 'glass_blur'; do
s=5;

CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
/home/sjtu/scratch/zltan/mmclassification/tools/my_tent_train.py \
/home/sjtu/scratch/zltan/mim/configs/vit-b-tent.py --launcher pytorch --corruption $c --severity $s;
done
