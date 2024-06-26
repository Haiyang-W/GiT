_base_ = ['../../_base_/seg_default_runtime.py',
'./git_large.py'
]
global_bin = _base_.global_bin
base_img_size = _base_.base_img_size
backend_args = None
classes_num=37
sun_rgbd_semseg_cfgs =  dict(
    mode='semantic_segmentation',
    grid_resolution_perwin=[14, 14],
    samples_grids_eachwin=32,
    grid_interpolate=True,
    num_classes=classes_num,
    num_vocal=classes_num+1,
    total_num_vocal=classes_num+1,
    max_decoder_length=16,
    global_only_image=True)

sun_rgbd_semseg_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(672, 672), keep_ratio=False),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='SegLoadAnnotations', reduce_zero_label=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='semantic_segmentation', 
                                            head_cfg=dict(num_classes=classes_num,
                                                            num_vocal=classes_num+1,
                                                            dec_length=16,
                                                            dec_pixel_resolution=[4, 4],
                                                            arg_max_inference=True,
                                                            ignore_index=255),
                                            git_cfg=sun_rgbd_semseg_cfgs)),
    dict(type='PackSegInputs', meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 
                                          'reduce_zero_label', 'task_name', 'head_cfg', 'git_cfg'))]

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='SunRGBDDataset',
        data_root='data/sunrgbd/',
        data_prefix=dict(
            img_path='images/validation', seg_map_path='annotations/validation'),
        pipeline=sun_rgbd_semseg_test_pipeline))
test_pipeline = sun_rgbd_semseg_test_pipeline
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
