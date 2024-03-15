_base_ = ['../../_base_/seg_default_runtime.py',
'./git_huge.py'
]
global_bin = _base_.global_bin
base_img_size = _base_.base_img_size
backend_args = None
classes_num = 8
cityscapes_det_cfgs = dict(
    mode='detection',
    grid_resolution_perwin=[5, 5],
    samples_grids_eachwin=10,
    grid_interpolate=True,
    num_bins=global_bin, # TODO: double check
    num_classes=classes_num,
    num_vocal=global_bin+1+classes_num+1,
    total_num_vocal=global_bin+1+classes_num+1,
    max_decoder_length=5,
    global_only_image=True)

cityscapes_det_test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(base_img_size, base_img_size), keep_ratio=False),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='detection', 
                                            head_cfg=dict(num_classes=classes_num,
                                                        num_vocal=classes_num+1+global_bin+1,
                                                        num_bins=global_bin,
                                                        dec_length=5,
                                                        arg_max_inference=True),
                                            git_cfg=cityscapes_det_cfgs)),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'task_name', 'head_cfg', 'git_cfg')),]

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        ann_file='annotations/instancesonly_filtered_gtFine_val.json',
        data_prefix=dict(img='leftImg8bit/val/'),
        test_mode=True,
        return_classes=True,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=cityscapes_det_test_pipeline,
        backend_args=backend_args))
test_pipeline = cityscapes_det_test_pipeline
test_dataloader = val_dataloader

val_evaluator = dict(
        type='CocoMetric',
        ann_file='data/cityscapes/' +
        'annotations/instancesonly_filtered_gtFine_val.json',
        metric='bbox',
        backend_args=backend_args)
test_evaluator = val_evaluator
