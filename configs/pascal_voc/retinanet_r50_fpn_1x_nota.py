_base_ = [
    '../_base_/models/retinanet_r50_fpn.py', '../_base_/datasets/nota.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_1x.py'
]
model = dict(bbox_head=dict(num_classes=5))
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
