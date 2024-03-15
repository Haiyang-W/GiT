_base_ = ['../../_base_/seg_default_runtime.py',
'./git_base.py'
]
global_bin = _base_.global_bin
base_img_size = _base_.base_img_size
ray_num = _base_.ray_num
backend_args = None
classes_num = 8
cityscapes_ins_cfgs = dict(
    mode='instance_segmentation',
    grid_resolution_perwin=[5, 5],
    samples_grids_eachwin=10,
    grid_interpolate=True,
    num_classes=classes_num,
    num_vocal=global_bin+1+classes_num+1,
    total_num_vocal=global_bin+1+classes_num+1,
    ray_num=ray_num,
    max_decoder_length=1+4+2+ray_num, # 24 is ray num
    global_only_image=True)

cityscapes_ins_test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, backend_args=backend_args),
    dict(type='Resize', scale=(base_img_size, base_img_size), keep_ratio=False),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='instance_segmentation', 
                                            head_cfg=dict(num_classes=classes_num,
                                                            num_vocal=classes_num+1+global_bin+1,
                                                            num_bins=global_bin,
                                                            dec_length=1+4+2+ray_num,
                                                            arg_max_inference=True,
                                                            ray_num=ray_num,
                                                            use_mass_center=True),
                                            git_cfg=cityscapes_ins_cfgs)),
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
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        return_classes=True,
        pipeline=cityscapes_ins_test_pipeline,
        backend_args=backend_args))
test_pipeline = cityscapes_ins_test_pipeline
test_dataloader = val_dataloader

val_evaluator =  dict(
        type='CocoMetric',
        ann_file='data/cityscapes/' +
        'annotations/instancesonly_filtered_gtFine_val.json',
        metric=['bbox', 'segm'],
        backend_args=backend_args)
test_evaluator = val_evaluator
