# dataset settings
dataset_type = 'ImageNetC'
data_prefix = '/run/determined/workdir/datasets/imagenet-c'
corruption = 'elastic_transform'
severity = 5

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
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