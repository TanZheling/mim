_base_ = [
    './_base_/custom.py',
    './_base_/model/vit-b.py', 
    './_base_/dataset/imagenetc384.py',
    './_base_/default_runtime.py'
]

# dataset settings
dataset_type = 'ImageNetC'
data_prefix = '/home/sjtu/dataset/ImageNet-C'
corruption = 'motion_blur'
severity = 5

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(384, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=384),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(384, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=384),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    shuffle=True,
    train=dict(
        type=dataset_type,
        data_prefix=data_prefix,
        pipeline=train_pipeline,
        corruption=corruption,
        severity=severity),
    val=dict(
        type=dataset_type,
        data_prefix=data_prefix,
        pipeline=test_pipeline,
        corruption=corruption,
        severity=severity),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix=data_prefix,
        pipeline=test_pipeline,
        corruption=corruption,
        severity=severity))
evaluation = dict(interval=1, metric='accuracy')

model = dict(backbone=dict(img_size=384))

corruption = 'fog'
severity = 5
batch_size = 12
gpu = 1
data = dict(samples_per_gpu=batch_size)



#custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]



# test-time setting
mode = ['entropy', 'contrast', 'cls'][0]
aug_type = ['NoAug', 'moco1Aug', 'selfT.2.10'][0]
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
        requires_grad=True,
        topk=(1,),
        cal_acc=True
    )
)

noaug = []


aug_dict = {
    'NoAug': noaug,
}

key_pipeline = aug_dict[aug_type] + [
    dict(type='Resize', size=(384, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=384),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

# optimizer
lr=0.000005
#optimizer = dict(type='SGD', lr=lr, momentum=0.9,weight_decay=1e-4)

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
max_epoch=1
cos=1e-9
lr_config = dict(policy='CosineAnnealing', min_lr=cos)
runner = dict(type='epochBasedRunner', max_epochs=max_epoch)

checkpoint_config = dict(interval=20)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='384ent',
                entity='zlt', 
                name='vit-b-bs{}-lr{}-gpu{}-repeat{}'.format(batch_size,lr,gpu,max_epoch)
            )
        )
    ]
)
#load_from = '/run/determined/workdir/scratch/bishe/pretrained_model/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth'
#load_from = '/run/determined/workdir/scratch/bishe/pretrained_model/INTERN_models/vit-b.pth'
#load_from = '/home/sjtu/scratch/zltan/pretrained_models/INTERN_models/vit-b.pth'
#load_from = '/home/sjtu/scratch/zltan/pretrained_models/timm_models/vit-b.pth'
load_from = '/home/sjtu/scratch/zltan/pretrained_models/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth'