# model settings
type = 'Retinanet'
model = dict(
    type='NASRetinaNet',
    pretrained=dict(
        use_load=True,
        load_path='./seed_mbv2.pt',        
        seed_num_layers=[1, 1, 2, 3, 4, 3, 3, 1, 1] # mbv2
        ),
    backbone=dict(
        type='RetinaNetBackbone',
        search_params=dict(
            sample_policy='prob', # prob uniform
            weight_sample_num=1,
            affine=False,
            track=False,
            net_scale=dict(
                chs = [32, 16, 24, 32, 64, 96, 160, 320],
                num_layers = [4, 4, 4, 4, 4, 1],
                strides = [2, 1, 2, 2, 2, 1, 2, 1, 1],
            ),
            primitives_normal=['k3_e3',
                                'k3_e6',
                                'k5_e3',
                                'k5_e6',
                                'k7_e3',
                                'k7_e6',
                                'skip',],
            primitives_reduce=['k3_e3',
                                'k3_e6',
                                'k5_e3',
                                'k5_e6',
                                'k7_e3',
                                'k7_e6',],
        ),
        output_indices=[2, 3, 5, 7],

    ),
    neck=dict(
        type='FPN',
        in_channels=[24, 32, 96, 320],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=81,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        octave_base_scale=4,
        scales_per_octave=3,
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[8, 16, 32, 64, 128],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0]))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    smoothl1_beta=0.11,
    gamma=2.0,
    alpha=0.25,
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)
# dataset settings
dataset_type = 'CocoDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file='annotations/instances_train2017.json',
        img_prefix='train2017/',
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=False,
        with_label=True),
    val=dict(
        img_prefix='train2017/',
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file='annotations/instances_val2017.json',
        img_prefix='val2017/',
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(
    weight_optim = dict(
        optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
        optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
    ),
    arch_optim = dict(
        optimizer = dict(type='Adam', lr=0.0003, weight_decay=0.001,
                        betas=(0.5, 0.999)),
        optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
    )
)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11, 14])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=500,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# configs for sub_obj optimizing
sub_obj=dict(
    if_sub_obj=True,
    type='flops',
    log_base=10.,
    sub_loss_factor=0.02
)
# yapf:enable
# runtime settings
total_epochs = 14

use_syncbn = False

arch_update_epoch = 8
alter_type = 'step' # step / epoch
train_data_ratio = 0.5
image_size_madds = (800, 1088)
model_info_interval = 1000
device_ids = range(8)
dist_params = dict(backend='nccl')
# log_level = 'DEBUG'
log_level = 'INFO'
work_dir = './work_dirs/'
load_from = None
resume_from = None
workflow = [('arch', 1),('train', 1)]
