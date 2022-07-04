_base_ = [
    '../_base_/models/retinanet_r50_fpn.py', '../_base_/datasets/nota.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_1x.py'
]
# model settings
norm_cfg = dict(type='BN', requires_grad=True)
#B3
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b3_3rdparty_8xb32-aa_in1k_20220119-5b4887a0.pth'
#B0
#checkpoint = 'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b0_3rdparty_8xb32-aa_in1k_20220119-8d939117.pth'

model = dict(
    type='RetinaNet',
    backbone=dict(
        _delete_=True,
        type='EfficientNet',
        arch='b3',
        drop_path_rate=0.2,
        out_indices=(3, 4, 5),
        frozen_stages=0,
        norm_cfg=dict(
            type='BN', requires_grad=True, eps=1e-3, momentum=0.01),
        norm_eval=False,
        init_cfg=dict(
            type='Pretrained', prefix='backbone', checkpoint=checkpoint)),
    neck=dict(
        in_channels=[48, 136, 384],#forB3
        #in_channels=[40, 112, 320],#forB0
        start_level=0,
        out_channels=256,
        relu_before_extra_convs=True,
        no_norm_on_lateral=True,
        norm_cfg=norm_cfg),
    bbox_head=dict(type='RetinaSepBNHead', num_classes=5, num_ins=5, norm_cfg=norm_cfg),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))


# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.001,
    step=[8, 11])
# learning policy
# actual epoch = 3 * 3 = 9
#lr_config = dict(policy='step', step=[3])
# runtime settings
#runner = dict(
#    type='EpochBasedRunner', max_epochs=4)  # actual epoch = 4 * 3 = 12
