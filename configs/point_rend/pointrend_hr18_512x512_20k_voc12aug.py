_base_ = [
    '../_base_/models/pointrend_hr18.py',
    '../_base_/datasets/pascal_voc12_aug.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(decode_head=[
    dict(
        type='FPNHead',
        in_channels=[
            sum([18, 36, 72, 144]),
            sum([18, 36, 72, 144]),
            sum([18, 36, 72, 144]),
            sum([18, 36, 72, 144])
        ],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=sum([18, 36, 72, 144]) / 2,
        dropout_ratio=-1,
        num_classes=21,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    dict(
        type='PointHead',
        in_channels=[sum([18, 36, 72, 144]) / 2],
        in_index=[0],
        channels=256,
        num_fcs=3,
        coarse_pred_each_layer=True,
        dropout_ratio=-1,
        num_classes=21,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
])
lr_config = dict(warmup='linear', warmup_iters=200)
