# dataset settings
dataset_type = 'CIFAR10C'
data_prefix = '/run/determined/workdir/datasets/cifar10'
img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=False)
train_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]


corruption = 'gaussian_noise'
severity = 5
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    sampler=dict(),
    shuffle=False,  # train shuffle, same as 'tent' repo
    train=dict(
        type=dataset_type,
        data_prefix=data_prefix,
        pipeline=train_pipeline,
        corruption=corruption,
        severity=severity,
        test_mode=True
    ),
    val=dict(
        type=dataset_type,
        data_prefix=data_prefix,
        pipeline=test_pipeline,
        corruption=corruption,
        severity=severity,
        test_mode=True
    ),
    test=dict(
        type=dataset_type,
        data_prefix=data_prefix,
        pipeline=test_pipeline,
        corruption=corruption,
        severity=severity,
        test_mode=True
    )
)