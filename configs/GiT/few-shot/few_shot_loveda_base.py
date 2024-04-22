_base_ = ['../../_base_/seg_default_runtime.py',
'./git_base.py'
]
load_from = './universal_base.pth'

global_bin = _base_.global_bin
base_img_size = _base_.base_img_size
backend_args = None
loveda_semseg_cfgs = dict(
    mode='semantic_segmentation',
    grid_resolution_perwin=[14, 14],
    samples_grids_eachwin=32,
    grid_interpolate=True,
    num_classes=7,
    num_vocal=7+1,
    total_num_vocal=7+1,
    max_decoder_length=16,
    global_only_image=True)

loveda_semseg_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='SegLoadAnnotations',reduce_zero_label=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='semantic_segmentation', 
                                            head_cfg=dict(num_classes=7,
                                                            num_vocal=8,
                                                            dec_length=16,
                                                            dec_pixel_resolution=[4, 4],
                                                            arg_max_inference=True,
                                                            ignore_index=255),
                                            git_cfg=loveda_semseg_cfgs)),
    dict(type='RandomChoice',
        transforms=[[dict(type='RandomChoiceResize', scales=[(672, 672)], keep_ratio=False)],
                    [dict(type='RandomChoiceResize', scales=[(int(672 * x * 0.1), int(672 * x * 0.1))  for x in range(10, 21)], keep_ratio=False),
                     dict(type='SegRandomCrop', crop_size=(672, 672), cat_max_ratio=0.75),]]),
    dict(type='MMCVRandomFlip', prob=0.5),
    dict(type='SegPhotoMetricDistortion'),
    dict(type='PackSegInputs', meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 
                                          'reduce_zero_label', 'task_name', 'head_cfg', 'git_cfg'))]
loveda_semseg_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(672, 672), keep_ratio=False),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='SegLoadAnnotations',reduce_zero_label=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='semantic_segmentation', 
                                            head_cfg=dict(num_classes=7,
                                                            num_vocal=8,
                                                            dec_length=16,
                                                            dec_pixel_resolution=[4, 4],
                                                            arg_max_inference=True,
                                                            ignore_index=255),
                                            git_cfg=loveda_semseg_cfgs)),
    dict(type='PackSegInputs', meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 
                                          'reduce_zero_label', 'task_name', 'head_cfg', 'git_cfg'))]

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=False),
    dataset=dict(
                type='LoveDADataset',
                data_root='data/loveDA',
                data_prefix=dict(
                    img_path='img_dir/train',
                    seg_map_path='ann_dir/train'),
                support_num=5*7,
                return_classes=True,
                pipeline=loveda_semseg_train_pipeline)   
    )


max_iters=100
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=max_iters, val_interval=50)
test_cfg = dict(type='TestLoop')
val_cfg = dict(type='ValLoop')

param_scheduler = [
    dict(type='MultiStepLR', by_epoch=False, milestones=[max_iters], gamma=0.1)
]


val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='LoveDADataset',
        data_root='data/loveDA',
        data_prefix=dict(
            img_path='img_dir/val',
            seg_map_path='ann_dir/val'),
        return_classes=True,
        pipeline=loveda_semseg_test_pipeline))
test_pipeline = loveda_semseg_test_pipeline
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

test_evaluator = val_evaluator


vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=50, max_keep_ckpts=50),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook',draw=False,interval=10,show=False))

log_processor = dict(type='LogProcessor', window_size=4000, by_epoch=False)