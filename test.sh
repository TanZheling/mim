for l in 0.00000001 0.00000005 ; 
do

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$(dirname $0)/..":$PYTHONPATH python /home/sjtu/scratch/zltan/mmclassification/tools/my_tent_train.py /home/sjtu/scratch/zltan/mim/configs/vit-cls.py --lr $l --corruption impulse_noise --severity 5;
done