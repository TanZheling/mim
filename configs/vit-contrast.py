_base_ = [
    './_base_/custom.py',
    './_base_/model/vit-b.py', 
    './_base_/dataset/imagenetc.py',
    './_base_/default_runtime.py'
]

custom_imports = dict(imports=[
    'tools.epoch_based_runner',
    'tools.pipeline_loading',
    'tools.linear_cls_head', 
  #  'tools.softmax_entropy_loss',
    #'tools.cifar',
    'tools.tent',
    'tools.convnext',
    'tools.vit',
    'tools.vision_transformer_head',
    'tools.image_classifier'
], allow_failed_imports=False)

batch_size = 16
class_num=1000

# test-time setting
mode = ['entropy', 'contrast', 'cls'][1]
repeat = 1
reset = [None, 'batch', 'sample'][0]

# contrast setting (proj)
# projector_dim = class_num
# queue_size = 32
# aug_type = ['NoAug', 'FlClAug', 'moco1Aug', 'selfT.2.10'][1]
# contrast_weight = 10
# temp = 30
# norm = ['L1Norm', 'L2Norm', 'softmax'][0]
# func = ['pgc', 'bankkPGC', 'best', 'bestAll', 'mocoCalib'][0]
# CLUE = False
# contrast setting (feat)
projector_dim = 2048
queue_size = 0
aug_type = ['NoAug', 'FlClAug', 'moco1Aug', 'selfT.2.10'][1]
contrast_weight = 10
temp = 30
norm = ['L1Norm', 'L2Norm', 'softmax'][1]
func = ['bank2PGC', 'best', 'bestAll', 'mocoCalib'][3]
CLUE = True

tag = ''

data = dict(
    samples_per_gpu=batch_size,  # batch_size = samples_per_gpu * num_gpu
    workers_per_gpu=4,
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

FlCl = [
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='RandomCrop', size=224, padding=4),
]

moco1 =  [
    dict(type='RandomCrop', size=224, padding=4),
    dict(type='RandomGrayscale', gray_prob=0.2),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
]

model = dict(
    type='imageClassifier',
    backbone=dict(
        type='VisionTransformerTent',
        arch='b',
        img_size=224,
        patch_size=16,
        drop_rate=0.1,
        init_cfg=[
            dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear')
        ]),
    neck=None,
    head=dict(
        type='visionTransformerClsHead',
        num_classes=1000,
        in_channels=768,
        requires_grad=False,
        loss=dict(type='SoftmaxEntropyLoss', loss_weight=1.0),
    ))

aug_dict = {
    'NoAug': [], 
    'FlClAug': FlCl, 
    'moco1Aug': moco1
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

test_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

# optimizer
optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    type='tentOptimizerHook',
    optimizer_cfg=optimizer,
    loss_cfg=dict(
        mode=mode,
        # contrast
        contrast_weight=contrast_weight,
        projector_dim=projector_dim,
        class_num=class_num,
        queue_size=queue_size,
        temp=temp / 100,
        model_cfg=model,
        norm=norm,
        origin_pipeline=test_pipeline,
        func=func,
        CLUE=CLUE
    ),
    grad_clip=None,
    reset=reset,
    repeat=repeat,
    augment_cfg=key_pipeline
)
# optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.0001)
runner = dict(type='epochBasedRunner', max_epochs=1)

checkpoint_config = dict(interval=100)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='layernorm',
                entity='zlt',
                name='_rednet26_b{}_cifar{}c_{}_{}_{}_R{}_r{}_q{}_p{}_w{}_{}{}'.format(
                    batch_size,
                    class_num,
                    mode,
                    func,
                    aug_type,
                    reset,
                    repeat,
                    queue_size,
                    projector_dim,
                    contrast_weight,
                    norm,
                    tag
                ),
            )
        )
    ]
)

load_from = '/home/sjtu/scratch/zltan/pretrained_models/timm_models/vit-b.pth'