for l in 0.1 0.05; 
do

CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$(dirname $0)/..":$PYTHONPATH python /home/sjtu/scratch/zltan/mmclassification/tools/my_tent_train.py /home/sjtu/scratch/zltan/mim/configs/vit-contrast.py --lr $l  --corruption fog --severity 5;
done