_base_ = ['../../_base_/seg_default_runtime.py',
'./git_base.py'
]
load_from = './universal_base.pth'

global_bin = _base_.global_bin
base_img_size = _base_.base_img_size
backend_args = None
widerface_det_cfgs = dict(
    mode='detection',
    grid_resolution_perwin=[5, 5],
    samples_grids_eachwin=10,
    grid_interpolate=True,
    num_bins=global_bin, # TODO: double check
    num_classes=1,
    num_vocal=global_bin+1+1+1,
    total_num_vocal=global_bin+1+1+1,
    max_decoder_length=5,
    global_only_image=True)

widerface_det_train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='detection', 
                                            head_cfg=dict(num_classes=1,
                                                        num_vocal=1+1+global_bin+1,
                                                        num_bins=global_bin,
                                                        dec_length=5,
                                                        arg_max_inference=True),
                                            git_cfg=widerface_det_cfgs)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomChoice',
        transforms=[[dict(type='RandomChoiceResize', scales=[(base_img_size, base_img_size)], keep_ratio=False)]]),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'task_name', 'head_cfg', 'git_cfg')),]

widerface_det_test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(base_img_size, base_img_size), keep_ratio=False),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='detection', 
                                            head_cfg=dict(num_classes=1,
                                                        num_vocal=1+1+global_bin+1,
                                                        num_bins=global_bin,
                                                        dec_length=5,
                                                        arg_max_inference=True),
                                            git_cfg=widerface_det_cfgs)),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'task_name', 'head_cfg', 'git_cfg')),]

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=False),
    dataset=dict(type='WIDERFaceDataset',
                data_root='data/WIDERFace/',
                ann_file='train.txt',
                data_prefix=dict(img='WIDER_train'),
                filter_cfg=dict(filter_empty_gt=True, bbox_min_size=17, min_size=32),
                pipeline=widerface_det_train_pipeline,
                support_num=5,
                return_classes=True,
                backend_args=backend_args)
    )

max_iters=50
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=max_iters, val_interval=50)
test_cfg = dict(type='TestLoop')
val_cfg = dict(type='ValLoop')

param_scheduler = [
    dict(type='MultiStepLR', by_epoch=False, milestones=[max_iters], gamma=0.1)
]


val_dataloader = dict(
    batch_size=3,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='WIDERFaceDataset',
        data_root='data/WIDERFace/',
        ann_file='val.txt',
        data_prefix=dict(img='WIDER_val'),
        test_mode=True,
        return_classes=True,
        pipeline=widerface_det_test_pipeline))
test_pipeline = widerface_det_test_pipeline
test_dataloader = val_dataloader

val_evaluator = dict(
    # TODO: support WiderFace-Evaluation for easy, medium, hard cases
    type='VOCMetric',
    metric='mAP',
    eval_mode='11points')

test_evaluator = val_evaluator


default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=50, max_keep_ckpts=50),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook',draw=False,interval=50,show=False))

log_processor = dict(type='LogProcessor', window_size=4000, by_epoch=False)