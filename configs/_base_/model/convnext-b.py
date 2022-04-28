# Model settings
model = dict(
    type='imageClassifier',
    backbone=dict(
        type='ConvNeXttent',
        arch='base',
        out_indices=(3, ),
        drop_path_rate=0.5,
        gap_before_final_norm=True,
        init_cfg=[
            dict(
                type='TruncNormal',
                layer=['Conv2d', 'Linear'],
                std=.02,
                bias=0.),
            dict(type='Constant', layer=['LayerNorm'], val=1., bias=0.),
        ]),
    head=dict(
        type='linearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(type='SoftmaxEntropyLoss', loss_weight=1.0),
    ))