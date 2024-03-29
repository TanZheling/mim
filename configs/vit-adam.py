_base_ = [
    './_base_/custom.py',
    './_base_/model/vit-b.py', 
    './_base_/dataset/imagenetc.py',
    './_base_/default_runtime.py'
]

corruption = 'defocus_blur'
severity = 5
batch_size=32
gpu=1
bs=batch_size*gpu
data = dict(samples_per_gpu=batch_size)



custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]



# test-time setting
mode = ['entropy', 'contrast', 'cls'][0]
aug_type = ['NoAug', 'FlCrAug', 'moco1Aug', 'selfT.2.10'][0]
repeat = 1
reset = [None, 'batch', 'sample'][0]

# entropy setting
entropy_weight = 1
entropy_type = ['entropy', 'infomax', 'memo'][0]
img_aug = ['weak', 'strong'][0]

att=True
fnn=True
norm=True
model = dict(
    backbone=dict(
        layer_cfgs=dict(fnn_grad=fnn,
                 att_grad=att,
                 norm_cfg=dict(type='LN',requires_grad=norm))
            
    ),
    head=dict(
        num_classes=1000,
        requires_grad=False,
        topk=(1,),
        cal_acc=True
    )
)

noaug = []

FlCr =  [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
]

aug_dict = {
    'NoAug': noaug,
    'FlCrAug': FlCr
}

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
'''
origin_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
'''
key_pipeline = aug_dict[aug_type] +  [
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]



# optimizer
lr=0.00000005
wd=0.0001
paramwise_cfg = dict(custom_keys={
    '.cls_token': dict(decay_mult=0.0),
    '.pos_embed': dict(decay_mult=0.0)
})
#optimizer = dict(type='SGD', lr=lr, momentum=0.9,weight_decay=1e-4)
optimizer = dict(
    type='AdamW',
    lr=lr,
    weight_decay=wd,
    paramwise_cfg=paramwise_cfg,
)
optimizer_config = dict(grad_clip=dict(max_norm=1.0))

# learning policy
"""lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=10000,
    warmup_ratio=1e-4,
)"""

optimizer_config = dict(
    type='TentOptimizerHook',
    optimizer_cfg=optimizer,
    loss_cfg=dict(
        mode=mode,
        entropy_weight=entropy_weight,
        entropy_type=entropy_type,
        img_aug=img_aug,
        #origin_pipeline=origin_pipeline
    ),
    grad_clip=None,
    reset=reset,
    repeat=repeat,
    augment_cfg=key_pipeline
)
max_epoch=10

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.0001)
runner = dict(type='epochBasedRunner', max_epochs=max_epoch)

checkpoint_config = dict(interval=20)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='normlayers',
                entity='zlt', 
                name='adam-intern-b-img-c-bs{}-lr{}-ep{}'.format(batch_size,lr,max_epoch)
            )
        )
    ]
)
#load_from = '/run/determined/workdir/scratch/bishe/pretrained_model/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth'
#load_from = '/run/determined/workdir/scratch/bishe/pretrained_model/INTERN_models/vit-b.pth'
#load_from = '/home/sjtu/scratch/zltan/pretrained_models/INTERN_models/vit-b.pth'
load_from = '/home/sjtu/scratch/zltan/pretrained_models/timm_models/vit-b.pth'