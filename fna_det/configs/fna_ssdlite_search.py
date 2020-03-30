# model settings

input_size = 300
model = dict(
    type='SingleStageDetectorSearch',
    pretrained=dict(
        use_load=True,
        load_path='./seed_mbv2.pt',
        seed_num_layers=[1, 1, 2, 3, 4, 3, 3, 1, 1] # mbv2
        ),
    backbone=dict(
        type='SSDLiteBackbone',
        input_size=input_size,
        search_params=dict(
            sample_policy='prob', # prob uniform
            weight_sample_num=1,
            affine=False,
            track=False,
            net_scale=dict(
                chs = [32, 16, 24, 32, 64, 96, 160, 320, 1280],
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
        output_indices=(6, 8),
    ),
    neck=None,
    bbox_head=dict(
        type='SSDLightHead',
        input_size=input_size,
        in_channels=(576, 1280, 512, 256, 256, 128),
        num_classes=81,
        anchor_strides=(16, 32, 64, 107, 160, 320),
        basesize_ratio_range=(0.2, 0.95),
        anchor_ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
        target_means=(.0, .0, .0, .0),
        target_stds=(0.1, 0.1, 0.2, 0.2),
        search=True))

# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0,
        ignore_iof_thr=-1,
        gt_max_assign_all=False),
    smoothl1_beta=1.,
    allowed_border=-1,
    pos_weight=-1,
    neg_pos_ratio=3,
    debug=False)
test_cfg = dict(
    # nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.02,
    nms=dict(type='nms', iou_thr=0.6),
    max_per_img=200)
# dataset settings
dataset_type = 'CocoDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
data = dict(
    imgs_per_gpu=16,
    workers_per_gpu=1,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file= 'annotations/instances_train2017.json',
            img_prefix= 'train2017/',
            img_scale=(320, 320),
            img_norm_cfg=img_norm_cfg,
            size_divisor=None,
            flip_ratio=0.5,
            with_mask=False,
            with_crowd=False,
            with_label=True,
            test_mode=False,
            extra_aug=dict(
                photo_metric_distortion=dict(
                    brightness_delta=32,
                    contrast_range=(0.5, 1.5),
                    saturation_range=(0.5, 1.5),
                    hue_delta=18),
                expand=dict(
                    mean=img_norm_cfg['mean'],
                    to_rgb=img_norm_cfg['to_rgb'],
                    ratio_range=(1, 4)),
                random_crop=dict(
                    min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3)),
            resize_keep_ratio=False)),
    val=dict(
        img_prefix='train2017/',
        img_scale=(320, 320),
        img_norm_cfg=img_norm_cfg,
        size_divisor=None,
        flip_ratio=0,
        with_mask=False,
        # with_crowd=False,
        with_label=True,
        # test_mode=True,
        resize_keep_ratio=False),
    test=dict(
        type=dataset_type,
        ann_file='annotations/instances_val2017.json',
        img_prefix='val2017/',
        img_scale=(320, 320),
        img_norm_cfg=img_norm_cfg,
        size_divisor=None,
        flip_ratio=0,
        with_mask=False,
        # with_crowd=False,
        with_label=False,
        test_mode=True,
        resize_keep_ratio=False))
# optimizer
optimizer = dict(
    weight_optim = dict(
        optimizer = dict(type='RMSprop', lr=0.05, eps=1.0, weight_decay=0.00004, momentum=0.9),
        optimizer_config = dict()
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
    step=[18, 25, 28]
    # target_lr=0.0001
    )
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# configs for sub_obj optimizing
sub_obj=dict(
    if_sub_obj=True,
    type='flops',
    log_base=10.,
    sub_loss_factor=0.15
)
type = 'SSDLite'
# yapf:enable
# runtime settings
total_epochs = 30

use_syncbn = False

arch_update_epoch = 15
alter_type = 'step' # step / epoch
train_data_ratio = 0.5
image_size_madds = (320, 320)
model_info_interval = 100
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/'
load_from = None
resume_from = None
workflow = [('arch', 1),('train', 5)]
