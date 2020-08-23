# model settings
model = dict(
    type='CaSe',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=3,
        strides=(1, 2, 1),
        dilations=(1, 1, 1),
        out_indices=(2,),
        frozen_stages=0,
        style='pytorch'),
    neck=dict(
        type='Conv',
        in_channels=1024,
        out_channels=256),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[4., 5.4, 7.2, 9.8, 13.2, 17.9, 24.2, 33.0, 44.1, 59.6, 80.0],
        anchor_ratios=[2.44],
        anchor_strides=[8],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
    ),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[8]),
    count_head=dict(
        type='CountHead',
        in_channels=256),
    similar_head=dict(
        type='SimilarHead',
        in_channels=256),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=2,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
    ),
)
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=6000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.00,
        nms=dict(type='cas_nms', thresh=0.5, t_2=1.5, t_1=1.0, N_st=1.5),
        max_per_img=100))
# dataset settings
dataset_type = 'CityDataset'
data_root = 'data/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'cityscapes/leftImg8bit/',
        img_scale=(2048, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True)
)
# yapf:enable
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '../work_dirs/case'
load_from = None
resume_from = None
workflow = [('train', 1)]
