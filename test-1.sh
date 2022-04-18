for l in 'jpeg_compression' 'shot_noise' ; 
do

CUDA_VISIBLE_DEVICES=2 PYTHONPATH="$(dirname $0)/..":$PYTHONPATH python /home/sjtu/scratch/zltan/mmclassification/tools/my_tent_train.py /home/sjtu/scratch/zltan/mim/configs/vit-b-tent.py  --corruption $l --severity 5;
done