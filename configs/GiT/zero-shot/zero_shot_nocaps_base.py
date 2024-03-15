_base_ = ['../../_base_/seg_default_runtime.py',
'./git_base.py'
]
global_bin = _base_.global_bin
base_img_size = _base_.base_img_size
backend_args = None

caption_cfgs = dict(
    mode='caption',
    grid_resolution_perwin=[1, 1],
    samples_grids_eachwin=1,
    use_checkpoints=True,
    grid_interpolate=False,
    loss_out_layer=[17], # 17 layers
    num_vocal=30524,
    total_num_vocal=30524,
    max_decoder_length=20,
    global_only_image=False,
    ignore_index=-100,
)

caption_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='caption', 
                                            head_cfg=dict(num_classes=30524,
                                                            num_vocal=30524,
                                                            dec_length=20,
                                                            arg_max_inference=True,
                                                            ignore_index=-100,
                                                            beam_num=3),
                                            git_cfg=caption_cfgs)),
    dict(type='Resize', scale=(224, 224), interpolation='bicubic', backend='pillow'),
    dict(type='PackInputs', meta_keys=['image_id', 'img_shape', 'task_name', 'head_cfg', 'git_cfg']),
]

val_dataloader = dict(
    batch_size=16,
    num_workers=5,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='NoCaps',
        data_root='data/nocaps/',
        data_prefix=dict(img_path='images/'),
        ann_file='annotations/nocaps_val_4500_captions.json',
        pipeline=caption_test_pipeline,
    ))
test_pipeline = caption_test_pipeline
test_dataloader = val_dataloader

val_evaluator = dict(
    type='COCOCaption',
    ann_file='data/nocaps/annotations/nocaps_val_4500_captions.json',
)
test_evaluator = val_evaluator
