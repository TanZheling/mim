for c in 'defocus_blur'	'pixelate' 'glass_blur' \
	        'elastic_transform'	'brightness' 'fog' 'contrast' \
    	    'frost'	'impulse_noise'	'jpeg_compression' 'shot_noise' \
            'zoom_blur'	'gaussian_noise' 'motion_blur' 'snow'; do
s=5;

CUDA_VISIBLE_DEVICES=2 PYTHONPATH="$(dirname $0)/..":$PYTHONPATH python /home/sjtu/scratch/zltan/mmclassification/tools/my_tent_train.py /home/sjtu/scratch/zltan/mim/configs/vit-in.py --corruption $c --severity $s;
done