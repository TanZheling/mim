# model settings
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
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1,
            mode='classy_vision'),
    ))
