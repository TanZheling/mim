for l in 'motion_blur' 'snow' ; 
do

CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$(dirname $0)/..":$PYTHONPATH python /home/sjtu/scratch/zltan/mmclassification/tools/my_tent_train.py /home/sjtu/scratch/zltan/mim/configs/vit-b-tent.py  --corruption $l --severity 5;
done