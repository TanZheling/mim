for c in 'defocus_blur'	'pixelate' 'glass_blur' \
	        'elastic_transform'	'brightness' 'fog' 'contrast' \
    	    'frost'	'impulse_noise'	'jpeg_compression' 'shot_noise' \
            'zoom_blur'	'gaussian_noise' 'motion_blur' 'snow'; 
do
s=5;
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$(dirname $0)/..":$PYTHONPATH python /run/determined/workdir/scratch/mmclassification/tools/my_tent_train.py /run/determined/workdir/scratch/mim/configs/vit-b-tent.py --corruption $c --severity $s;
done