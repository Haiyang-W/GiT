_base_ = ['../_base_/seg_default_runtime.py']
pretrained = 'https://download.openmmlab.com/mmclassification/v1/vit_sam/vit-base-p16_sam-pre_3rdparty_sa1b-1024px_20230411-2320f9cc.pth'

# hyper parameter for each tasks
caption_cfgs = dict(
    mode='caption',
    grid_resolution_perwin=[1, 1],
    samples_grids_eachwin=1,
    use_checkpoints=True,
    grid_interpolate=False,
    num_vocal=30524,
    max_decoder_length=20,
    global_only_image=False,
    ignore_index=-100)

model = dict(
    type='GiT',
    support_tasks=['detection', 'semantic_segmentation', 'instance_segmentation', 'caption', 'grounding'],
    use_checkpoints=True,
    tokenizer=dict(type='BlipTokenizer', name_or_path='bert-base-uncased'),
    bert_embed=dict(type='bert-base', hidden_size=768, pretrain_path='./bert_embed.pt'),
    data_preprocessor=dict(
        type='GeneralDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_seg=True,
        seg_pad_value=255,
        pad_size_divisor=224),
    backbone=dict(
        type='ViTGiT',
        arch='base',
        img_size=1120, # 1120 match the resolution used in multi-task
        patch_size=16,
        out_channels=0,
        use_abs_pos=True,
        use_rel_pos=True,
        window_size=14,
        out_type='featmap',
        use_checkpoints=True,
        new_more_layers=['win', 'win', 'win', 'win', 'win', 'win'],  # win, global
        drop_path_rate=0.1,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained, prefix='backbone.'),),
    head_list=dict( 
        # non parametric task-specific heads
        caption_head=dict(type='GiTCaptionHead')),
    )

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
# pipeline for image caption
caption_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='caption', 
                                            head_cfg=dict(num_classes=30524,
                                                            num_vocal=30524,
                                                            dec_length=20,
                                                            arg_max_inference=True,
                                                            ignore_index=-100,
                                                            beam_num=2),
                                            git_cfg=caption_cfgs)),
    dict(type='RandomResizedCrop', scale=224, interpolation='bicubic', backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='CleanCaption', keys='gt_caption'),
    dict(type='PackInputs', algorithm_keys=['gt_caption'], meta_keys=['image_id','img_shape', 'task_name', 'head_cfg', 'git_cfg'],),
]

caption_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='caption', 
                                            head_cfg=dict(num_classes=30524,
                                                            num_vocal=30524,
                                                            dec_length=20,
                                                            ignore_index=-100,
                                                            beam_num=2),
                                            git_cfg=caption_cfgs)),
    dict(type='Resize', scale=(224, 224), interpolation='bicubic', backend='pillow'),
    dict(type='PackInputs', meta_keys=['image_id', 'img_shape', 'task_name', 'head_cfg', 'git_cfg']),
]
train_dataloader = dict(
    batch_size=3,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='GroupMultiSourceNonMixedSampler', batch_size=3, source_ratio=[1.], 
                 if_group=[False], shuffle=True),
    batch_sampler=None,
    dataset=dict(type='ConcatDataset',
                 ignore_keys=['reduce_zero_label', 'label_map', 'classes', 'palette'],
        datasets=[
            dict(type='COCOCaption',
                data_root='data/coco_2014',
                ann_file='annotations/coco_karpathy_train.json',
                pipeline=caption_train_pipeline),
            ]),      
    )

test_pipeline = caption_test_pipeline
val_dataloader = dict(batch_size=16,
        num_workers=2,
        persistent_workers=True,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='COCOCaption',
            data_root='data/coco_2014',
            ann_file='annotations/coco_karpathy_test.json',
            pipeline=caption_test_pipeline))
test_dataloader = val_dataloader

# optimizer
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
        }))

val_evaluator = dict(type='COCOCaption',
        ann_file='data/coco_2014/annotations/coco_karpathy_test_gt.json',)
test_evaluator = val_evaluator

# learning policy
max_iters=120000
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=max_iters, val_interval=5000)
test_cfg = dict(type='TestLoop')
val_cfg = dict(type='ValLoop')

param_scheduler = [dict(
          type='CosineAnnealingLR',
          T_max=max_iters,
          eta_min=2e-6,
          begin=0,
          end=max_iters,
          by_epoch=False,)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (3 samples per GPU)
auto_scale_lr = dict(base_batch_size=24)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000, max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

log_processor = dict(type='LogProcessor', window_size=4000, by_epoch=False)
