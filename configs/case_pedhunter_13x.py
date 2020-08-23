# model settings
model = dict(
    type='CaSe',
    backbone=dict(
        type='VGG16',
        depth=16),
    neck=None,
    rpn_head=dict(
        type='RPNHead',
        in_channels=512,
        feat_channels=512,
        anchor_scales=[5.2, 7.02, 9.36, 12.74, 17.16, 23.27, 31.46, 42.9, 57.33, 77.48, 104.],
        anchor_ratios=[2.44],
        anchor_strides=[8],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
    ),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=512,
        featmap_strides=[8]),
    count_head=dict(
        type='CountHead'),
    similar_head=dict(
        type='SimilarHead'),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=2,
        in_channels=512,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=2,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
    ),
    mask_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2),
        out_channels=512,
        featmap_strides=[8]),
    mask_head=dict(
        type='FCNMaskHead',
        num_convs=4,
        in_channels=512,
        conv_out_channels=256,
        class_agnostic=True,
        num_classes=2),
)
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=12000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.00,
        mask_thr_binary=0.5,
        nms=dict(type='cas_nms', thresh=0.5, t_2=1.5, t_1=1.0, N_st=1.5),
        max_per_img=100))
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
        img_scale=(2688, 1344),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True)
)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '../work_dirs/case'
load_from = None
resume_from = None
workflow = [('train', 1)]
