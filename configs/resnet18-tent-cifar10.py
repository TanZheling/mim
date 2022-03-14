_base_ = [
    './_base_/custom.py',
    './_base_/model/resnet18_cifar.py', 
    './_base_/dataset/cifar10c_bs16.py',
    './_base_/default_runtime.py'
]
data = dict(
    samples_per_gpu=128, # samples_per_gpu=64 for 4 gpus
    workers_per_gpu=2,
)

# test-time setting
mode = ['entropy', 'contrast', 'cls'][0]
aug_type = ['NoAug', 'FlCrAug', 'moco1Aug', 'selfT.2.10'][0]
repeat = 1
reset = [None, 'batch', 'sample'][0]

# entropy setting
entropy_weight = 1
entropy_type = ['entropy', 'infomax', 'memo'][0]
img_aug = ['weak', 'strong'][0]


model = dict(
    backbone=dict(
        conv_cfg=dict(type='Conv', requires_grad=False),
        norm_cfg=dict(type='BN', requires_grad=True),
    ),
    head=dict(
        num_classes=10, topk=(1,),
        loss=dict(type='SoftmaxEntropyLoss', loss_weight=1.0),
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

key_pipeline = aug_dict[aug_type] + [
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

# optimizer
optimizer = dict(type='Adam', lr=1e-5, betas=(0.9, 0.999), weight_decay=0)

optimizer_config = dict(
    type='TentOptimizerHook',
    optimizer_cfg=optimizer,
    loss_cfg=dict(
        mode=mode,
        entropy_weight=entropy_weight,
        entropy_type=entropy_type,
        img_aug=img_aug
    ),
    grad_clip=None,
    reset=reset,
    repeat=repeat,
    augment_cfg=key_pipeline
)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.0001)
runner = dict(type='epochBasedRunner', max_epochs=1)

checkpoint_config = dict(interval=20)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='transformer',
                entity='zlt', 
                name='tent-res18-cifar10'
            )
        )
    ]
)

load_from = '/run/determined/workdir/scratch/mmclassification/mywork/resnet18.pth'