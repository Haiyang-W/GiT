_base_ = ['../../_base_/seg_default_runtime.py',
'./git_base.py'
]
load_from = './universal_base.pth'

global_bin = _base_.global_bin
base_img_size = _base_.base_img_size
backend_args = None
deepfashion_det_cfgs = dict(
    mode='detection',
    grid_resolution_perwin=[5, 5],
    samples_grids_eachwin=10,
    grid_interpolate=True,
    num_bins=global_bin, # TODO: double check
    num_classes=15,
    num_vocal=global_bin+1+15+1,
    total_num_vocal=global_bin+1+15+1,
    max_decoder_length=5,
    global_only_image=True)

deepfashion_det_train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='detection', 
                                            head_cfg=dict(num_classes=15,
                                                        num_vocal=15+1+global_bin+1,
                                                        num_bins=global_bin,
                                                        dec_length=5,
                                                        arg_max_inference=True),
                                            git_cfg=deepfashion_det_cfgs)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomChoice',
        transforms=[[dict(type='RandomChoiceResize', scales=[(base_img_size, base_img_size)], keep_ratio=False)]]),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'task_name', 'head_cfg', 'git_cfg')),]

deepfashion_det_test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(base_img_size, base_img_size), keep_ratio=False),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='detection', 
                                            head_cfg=dict(num_classes=15,
                                                        num_vocal=15+1+global_bin+1,
                                                        num_bins=global_bin,
                                                        dec_length=5,
                                                        arg_max_inference=True),
                                            git_cfg=deepfashion_det_cfgs)),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'task_name', 'head_cfg', 'git_cfg')),]

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=False),
    dataset=dict(type='DeepFashionDataset',
                data_root='data/DeepFashion/In-shop/',
                ann_file='Anno/segmentation/DeepFashion_segmentation_train.json',
                data_prefix=dict(img='Img/'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                support_num=15*5,
                pipeline=deepfashion_det_train_pipeline,
                return_classes=True,
                backend_args=backend_args)
    )

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.05),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'backbone.layers.6': dict(lr_mult=0.2286),
            'backbone.layers.7': dict(lr_mult=0.3571),
            'backbone.layers.8': dict(lr_mult=0.4858),
            'backbone.layers.9': dict(lr_mult=0.6143),
            'backbone.layers.10': dict(lr_mult=0.7429),
            'backbone.layers.11': dict(lr_mult=0.8714),
            'backbone.layers.12': dict(lr_mult=1.0),
            'backbone.layers.13': dict(lr_mult=1.0),
            'backbone.layers.14': dict(lr_mult=1.0),
            'backbone.layers.15': dict(lr_mult=1.0),
            'backbone.layers.16': dict(lr_mult=1.0),
            'backbone.layers.17': dict(lr_mult=1.0),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        })
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
    batch_size=3,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DeepFashionDataset',
        data_root='data/DeepFashion/In-shop/',
        ann_file='Anno/segmentation/DeepFashion_segmentation_query.json',
        data_prefix=dict(img='Img/'),
        test_mode=True,
        return_classes=True,
        pipeline=deepfashion_det_test_pipeline))
test_pipeline = deepfashion_det_test_pipeline
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/DeepFashion/In-shop/' +
    'Anno/segmentation/DeepFashion_segmentation_query.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

test_evaluator = val_evaluator


default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=50, max_keep_ckpts=50),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook',draw=False,interval=50,show=False))

log_processor = dict(type='LogProcessor', window_size=4000, by_epoch=False)