# dataset settings
dataset_type = 'NOTADataset'
data_root = 'data/facial_emotion_data/'
img_norm_cfg = dict(
    mean=[114.4893, 120.3964, 129.6415], std=[68.3687, 69.1964, 71.9992], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(224, 224), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(224, 224),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root+'trainval.txt',
            img_prefix=data_root+'train/',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
            ann_file=data_root+'trainval.txt',
            img_prefix=data_root+'train/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
            ann_file=data_root+'trainval.txt',
            img_prefix=data_root+'train/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mAP')
