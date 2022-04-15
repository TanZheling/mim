for l in 0.000005 ;
do


CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$(dirname $0)/..":$PYTHONPATH python /home/sjtu/scratch/zltan/mmclassification/tools/my_tent_train.py /home/sjtu/scratch/zltan/mim/configs/vit-b-tent.py --lr $l --corruption impulse_noise --severity 5;
done