_base_ = ['../_base_/seg_default_runtime.py']
backend_args = None
pretrained = 'https://download.openmmlab.com/mmclassification/v1/vit_sam/vit-huge-p16_sam-pre_3rdparty_sa1b-1024px_20230411-3f13c653.pth'
global_bin = 2240
# hyper parameter for each tasks
# detection: object365, openimage, lvisv1, VOC0712, nuimages, coco
obj365_det_cfgs = dict(
    mode='detection',
    grid_resolution_perwin=[5, 5],
    samples_grids_eachwin=10,
    grid_interpolate=True,
    num_bins=global_bin, # TODO: double check
    num_classes=365,
    num_vocal=global_bin+1+365+1,
    total_num_vocal=global_bin+1+365+1,
    max_decoder_length=5,
    global_only_image=True)

openimage_det_cfgs = dict(
    mode='detection',
    grid_resolution_perwin=[5, 5],
    samples_grids_eachwin=10,
    grid_interpolate=True,
    num_bins=global_bin, # TODO: double check
    num_classes=601,
    num_vocal=global_bin+1+601+1,
    total_num_vocal=global_bin+1+601+1,
    max_decoder_length=5,
    global_only_image=True)

lvisv1_det_cfgs = dict(
    mode='detection',
    grid_resolution_perwin=[5, 5],
    samples_grids_eachwin=10,
    grid_interpolate=True,
    num_bins=global_bin, # TODO: double check
    num_classes=1203,
    num_vocal=global_bin+1+1203+1,
    total_num_vocal=global_bin+1+1203+1,
    max_decoder_length=5,
    global_only_image=True)


voc0712_det_cfgs = dict(
    mode='detection',
    grid_resolution_perwin=[5, 5],
    samples_grids_eachwin=10,
    grid_interpolate=True,
    num_bins=global_bin, # TODO: double check
    num_classes=20,
    num_vocal=global_bin+1+20+1,
    total_num_vocal=global_bin+1+20+1,
    max_decoder_length=5,
    global_only_image=True)

nuimage_det_cfgs = dict(
    mode='detection',
    grid_resolution_perwin=[5, 5],
    samples_grids_eachwin=10,
    grid_interpolate=True,
    num_bins=global_bin, # TODO: double check
    num_classes=10,
    num_vocal=global_bin+1+10+1,
    total_num_vocal=global_bin+1+10+1,
    max_decoder_length=5,
    global_only_image=True)

coco_det_cfgs = dict(
    mode='detection',
    grid_resolution_perwin=[5, 5],
    samples_grids_eachwin=10,
    grid_interpolate=True,
    num_bins=global_bin, # TODO: double check
    num_classes=80,
    num_vocal=global_bin+1+80+1,
    total_num_vocal=global_bin+1+80+1,
    max_decoder_length=5,
    global_only_image=True)

BDD100K_det_cfgs = dict(
    mode='detection',
    grid_resolution_perwin=[5, 5],
    samples_grids_eachwin=10,
    grid_interpolate=True,
    num_bins=global_bin, # TODO: double check
    num_classes=10,
    num_vocal=global_bin+1+10+1,
    total_num_vocal=global_bin+1+10+1,
    max_decoder_length=5,
    global_only_image=True)

# instance segmentation: lvis, openimage, nuimage, coco
ray_num=24

lvisv1_ins_cfgs = dict(
    mode='instance_segmentation',
    grid_resolution_perwin=[5, 5],
    samples_grids_eachwin=10, # TODO: change it
    grid_interpolate=True,
    num_classes=1203,
    num_vocal=global_bin+1+1203+1,
    total_num_vocal=global_bin+1+1203+1,
    ray_num=ray_num,
    max_decoder_length=1+4+2+24, # 24 is ray num
    global_only_image=True)

nuimage_ins_cfgs = dict(
    mode='instance_segmentation',
    grid_resolution_perwin=[5, 5],
    samples_grids_eachwin=10, # TODO: change it
    grid_interpolate=True,
    num_classes=10,
    num_vocal=global_bin+1+10+1,
    total_num_vocal=global_bin+1+10+1,
    ray_num=ray_num,
    max_decoder_length=1+4+2+24, # 24 is ray num
    global_only_image=True)

openimage_ins_cfgs = dict(
    mode='instance_segmentation',
    grid_resolution_perwin=[5, 5],
    samples_grids_eachwin=10, # TODO: change it
    grid_interpolate=True,
    num_classes=601,
    num_vocal=global_bin+1+601+1,
    total_num_vocal=global_bin+1+601+1,
    ray_num=ray_num,
    max_decoder_length=1+4+2+24, # 24 is ray num
    global_only_image=True)

coco_ins_cfgs = dict(
    mode='instance_segmentation',
    grid_resolution_perwin=[5, 5],
    samples_grids_eachwin=10, # TODO: change it
    grid_interpolate=True,
    num_classes=80,
    num_vocal=global_bin+1+80+1,
    total_num_vocal=global_bin+1+80+1,
    ray_num=ray_num,
    max_decoder_length=1+4+2+ray_num, # 24 is ray num
    global_only_image=True)

BDD100K_ins_cfgs = dict(
    mode='instance_segmentation',
    grid_resolution_perwin=[5, 5],
    samples_grids_eachwin=10,
    grid_interpolate=True,
    num_classes=8,
    num_vocal=global_bin+1+8+1,
    total_num_vocal=global_bin+1+8+1,
    ray_num=ray_num,
    max_decoder_length=1+4+2+ray_num, # 24 is ray num
    global_only_image=True)

# semantic segmentation: coco_stuff164k, nuimage_seg, Pascal Context, BDD100K, Mapillary V2,  ade20k
coco_stuff164k_semseg_cfgs = dict(
    mode='semantic_segmentation',
    grid_resolution_perwin=[14, 14],
    samples_grids_eachwin=32,
    grid_interpolate=True,
    num_classes=171,
    num_vocal=171+1,
    total_num_vocal=171+1,
    max_decoder_length=16,
    global_only_image=True)

nuimage_seg_semseg_cfgs = dict(
    mode='semantic_segmentation',
    grid_resolution_perwin=[14, 14],
    samples_grids_eachwin=32,
    grid_interpolate=True,
    num_classes=31,
    num_vocal=31+1,
    total_num_vocal=31+1,
    max_decoder_length=16,
    global_only_image=True)

pascal_context59_semseg_cfgs = dict(
    mode='semantic_segmentation',
    grid_resolution_perwin=[14, 14],
    samples_grids_eachwin=32,
    grid_interpolate=True,
    num_classes=59,
    num_vocal=59+1,
    total_num_vocal=59+1,
    max_decoder_length=16,
    global_only_image=True)

BDD100K_semseg_cfgs = dict(
    mode='semantic_segmentation',
    grid_resolution_perwin=[14, 14],
    samples_grids_eachwin=32,
    grid_interpolate=True,
    num_classes=19,
    num_vocal=19+1,
    total_num_vocal=19+1,
    max_decoder_length=16,
    global_only_image=True)

mapillary_v2_semseg_cfgs = dict(
    mode='semantic_segmentation',
    grid_resolution_perwin=[14, 14],
    samples_grids_eachwin=32,
    grid_interpolate=True,
    num_classes=124,
    num_vocal=124+1,
    total_num_vocal=124+1,
    max_decoder_length=16,
    global_only_image=True)

ade20k_semseg_cfgs = dict(
    mode='semantic_segmentation',
    grid_resolution_perwin=[14, 14],
    samples_grids_eachwin=32,
    grid_interpolate=True,
    num_classes=150,
    num_vocal=150+1,
    total_num_vocal=150+1,
    max_decoder_length=16,
    global_only_image=True)

# general caption cfg: CCS(CC3M+CC12M+SBU), Visual Genome, COCO Caption
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

# grounding cfg: refcoco, refcoco+, refcocog
grounding_cfgs = dict(
    mode='grounding',
    grid_resolution_perwin=[1, 1],
    samples_grids_eachwin=1,
    grid_interpolate=False,
    num_bins=448, # TODO: double check
    num_classes=0,
    num_vocal=448+1,
    total_num_vocal=448+1,
    max_decoder_length=4,
    global_only_image=False)

# just placeholder
num_bins = global_bin
num_classes = 80
num_vocal = (num_bins + 1) + num_classes + 1 # use in tasks

# real hyper
base_img_size = 1120
feat_dim = 768
patch_size = 16

model = dict(
    type='GiT',
    support_tasks=['detection', 'semantic_segmentation', 'instance_segmentation', 'caption', 'grounding'],
    use_checkpoints=True,
    tokenizer=dict(type='BlipTokenizer', name_or_path='bert-base-uncased'),
    bert_embed=dict(type='bert-base', hidden_size=1280, pretrain_path='./bert_embed_huge.pt'),
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
        arch='huge',
        img_size=base_img_size,
        patch_size=16,
        out_channels=0,
        use_abs_pos=True,
        use_rel_pos=True,
        window_size=14,
        out_type='featmap',
        out_indices=[0,1,2,3,4,5,6,7,8,9,10,11,12],
        use_checkpoints=True,
        new_more_layers=['win', 'win', 'win', 'win', 'win', 'win'],  # win, global
        drop_path_rate=0.4,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained, prefix='backbone.'),),
    head_list=dict(
        # non parametric task-specific heads
        detection_head=dict(type='GiTDetHead',
            train_cfg=dict(
                assigner=dict(
                    type='HungarianAssigner',
                    match_costs=[
                        dict(type='PointsL1Cost', weight=5.0, box_format='xywh'),
                    ])),
            test_cfg=dict(max_per_img=100)),
        instance_segmentation_head=dict(type='GiTInsSegHead',
            train_cfg=dict(
                assigner=dict(
                    type='HungarianAssigner',
                    match_costs=[
                        dict(type='PointsL1Cost', weight=5.0, box_format='xywh'),
                    ])),
            test_cfg=dict(max_per_img=100)),
        semantic_segmentation_head=dict(type='GiTSemSegHead'),
        caption_head=dict(type='GiTCaptionHead'),
        grounding_head=dict(type='GiTGroundingHead')),)

## pipeline for detection obj365 NOTE 170w images
obj365_det_train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='detection', 
                                            head_cfg=dict(num_classes=365,
                                                        num_vocal=365+1+global_bin+1,
                                                        num_bins=global_bin,
                                                        dec_length=5,
                                                        arg_max_inference=True),
                                            git_cfg=obj365_det_cfgs)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomChoice',
        transforms=[[dict(type='RandomChoiceResize', scales=[(base_img_size, base_img_size)], keep_ratio=False)]]),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'task_name', 'head_cfg', 'git_cfg')),]


## pipeline for detection openimage NOTE 220w images
openimages_det_train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='detection', 
                                            head_cfg=dict(num_classes=601,
                                                        num_vocal=601+1+global_bin+1,
                                                        num_bins=global_bin,
                                                        dec_length=5,
                                                        arg_max_inference=True),
                                            git_cfg=openimage_det_cfgs)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomChoice',
        transforms=[[dict(type='RandomChoiceResize', scales=[(base_img_size, base_img_size)], keep_ratio=False)]]),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'task_name', 'head_cfg', 'git_cfg')),]


## pipeline for detection lvis v1.0 NOTE 12w images
lvisv1_det_train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='detection', 
                                            head_cfg=dict(num_classes=1203,
                                                        num_vocal=1203+1+global_bin+1,
                                                        num_bins=global_bin,
                                                        dec_length=5,
                                                        arg_max_inference=True),
                                            git_cfg=lvisv1_det_cfgs)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomChoice',
        transforms=[[dict(type='RandomChoiceResize', scales=[(base_img_size, base_img_size)], keep_ratio=False)],
                     
                    [dict(type='RandomChoiceResize', scales=[(400, 4200), (500, 4200), (600, 4200)], keep_ratio=True),
                     dict(type='RandomCrop', crop_type='absolute_range', crop_size=(384, 600), allow_negative_crop=True),
                     dict(type='RandomChoiceResize', scales=[(base_img_size, base_img_size)], keep_ratio=False)]]),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'task_name', 'head_cfg', 'git_cfg')),]


voc0712_det_train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='detection', 
                                            head_cfg=dict(num_classes=20,
                                                        num_vocal=20+1+global_bin+1,
                                                        num_bins=global_bin,
                                                        dec_length=5,
                                                        arg_max_inference=True),
                                            git_cfg=voc0712_det_cfgs)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomChoice',
        transforms=[[dict(type='RandomChoiceResize', scales=[(base_img_size, base_img_size)], keep_ratio=False)]]),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'task_name', 'head_cfg', 'git_cfg')),]

nuimage_det_train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='detection', 
                                            head_cfg=dict(num_classes=10,
                                                        num_vocal=10+1+global_bin+1,
                                                        num_bins=global_bin,
                                                        dec_length=5,
                                                        arg_max_inference=True),
                                            git_cfg=nuimage_det_cfgs)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomChoice',
        transforms=[[dict(type='RandomChoiceResize', scales=[(base_img_size, base_img_size)], keep_ratio=False)]]),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'task_name', 'head_cfg', 'git_cfg')),]

coco_det_train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='detection', 
                                            head_cfg=dict(num_classes=num_classes,
                                                        num_vocal=num_vocal,
                                                        num_bins=num_bins,
                                                        dec_length=5,
                                                        arg_max_inference=True),
                                            git_cfg=coco_det_cfgs)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomChoice',
        transforms=[[dict(type='RandomChoiceResize', scales=[(base_img_size, base_img_size)], keep_ratio=False)],

                    [dict(type='RandomChoiceResize', scales=[(400, 4200), (500, 4200), (600, 4200)], keep_ratio=True),
                     dict(type='RandomCrop', crop_type='absolute_range', crop_size=(384, 600), allow_negative_crop=True),
                     dict(type='RandomChoiceResize', scales=[(base_img_size, base_img_size)], keep_ratio=False)]]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5)),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'task_name', 'head_cfg', 'git_cfg')),]

coco_det_test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(base_img_size, base_img_size), keep_ratio=False),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='detection', 
                                            head_cfg=dict(num_classes=num_classes,
                                                        num_vocal=num_vocal,
                                                        num_bins=num_bins,
                                                        dec_length=5,
                                                        arg_max_inference=True),
                                            git_cfg=coco_det_cfgs)),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'task_name', 'head_cfg', 'git_cfg')),]

BDD100K_det_train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='detection',
                                            head_cfg=dict(num_classes=10,
                                                        num_vocal=10+1+global_bin+1,
                                                        num_bins=global_bin,
                                                        dec_length=5,
                                                        arg_max_inference=True),
                                            git_cfg=BDD100K_det_cfgs)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomChoice',
        transforms=[[dict(type='RandomChoiceResize', scales=[(base_img_size, base_img_size)], keep_ratio=False)],

                    [dict(type='RandomChoiceResize', scales=[(400, 4200), (500, 4200), (600, 4200)], keep_ratio=True),
                     dict(type='RandomCrop', crop_type='absolute_range', crop_size=(384, 600), allow_negative_crop=True),
                     dict(type='RandomChoiceResize', scales=[(base_img_size, base_img_size)], keep_ratio=False)]]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5)),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'task_name', 'head_cfg', 'git_cfg')),]

## pipeline for instance segmentation lvis 1.0 NOTE 12w images
lvisv1_ins_train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='instance_segmentation',  
                                            head_cfg=dict(num_classes=1203,
                                                            num_vocal=1203+1+global_bin+1,
                                                            num_bins=global_bin,
                                                            dec_length=1+4+2+ray_num,
                                                            arg_max_inference=True,
                                                            ray_num=ray_num,
                                                            use_mass_center=True),
                                            git_cfg=lvisv1_ins_cfgs)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomChoice',
        transforms=[[dict(type='RandomChoiceResize', scales=[(base_img_size, base_img_size)], keep_ratio=False)],
                     
                    [dict(type='RandomChoiceResize', scales=[(400, 4200), (500, 4200), (600, 4200)], keep_ratio=True),
                     dict(type='RandomCrop', crop_type='absolute_range', crop_size=(384, 600), allow_negative_crop=True),
                     dict(type='RandomChoiceResize', scales=[(base_img_size, base_img_size)], keep_ratio=False)]]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'task_name', 'head_cfg', 'git_cfg')),
    dict(type='Mask2DenseContour', restore_all=True,)]

openimage_ins_train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='instance_segmentation',  
                                            head_cfg=dict(num_classes=601,
                                                            num_vocal=601+1+global_bin+1,
                                                            num_bins=global_bin,
                                                            dec_length=1+4+2+ray_num,
                                                            arg_max_inference=True,
                                                            ray_num=ray_num,
                                                            use_mass_center=True),
                                            git_cfg=openimage_ins_cfgs)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomChoice',
        transforms=[[dict(type='RandomChoiceResize', scales=[(base_img_size, base_img_size)], keep_ratio=False)],
                     
                    [dict(type='RandomChoiceResize', scales=[(400, 4200), (500, 4200), (600, 4200)], keep_ratio=True),
                     dict(type='RandomCrop', crop_type='absolute_range', crop_size=(384, 600), allow_negative_crop=True),
                     dict(type='RandomChoiceResize', scales=[(base_img_size, base_img_size)], keep_ratio=False)]]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'task_name', 'head_cfg', 'git_cfg')),
    dict(type='Mask2DenseContour', restore_all=True,)]

nuimage_ins_train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='instance_segmentation',  
                                            head_cfg=dict(num_classes=10,
                                                            num_vocal=10+1+global_bin+1,
                                                            num_bins=global_bin,
                                                            dec_length=1+4+2+ray_num,
                                                            arg_max_inference=True,
                                                            ray_num=ray_num,
                                                            use_mass_center=True),
                                            git_cfg=nuimage_ins_cfgs)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomChoice',
        transforms=[[dict(type='RandomChoiceResize', scales=[(base_img_size, base_img_size)], keep_ratio=False)],
                     
                    [dict(type='RandomChoiceResize', scales=[(400, 4200), (500, 4200), (600, 4200)], keep_ratio=True),
                     dict(type='RandomCrop', crop_type='absolute_range', crop_size=(384, 600), allow_negative_crop=True),
                     dict(type='RandomChoiceResize', scales=[(base_img_size, base_img_size)], keep_ratio=False)]]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'task_name', 'head_cfg', 'git_cfg')),
    dict(type='Mask2DenseContour', restore_all=True,)]

coco_ins_train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='instance_segmentation', 
                                            head_cfg=dict(num_classes=num_classes,
                                                            num_vocal=num_vocal,
                                                            num_bins=num_bins,
                                                            dec_length=1+4+2+ray_num,
                                                            arg_max_inference=True,
                                                            ray_num=ray_num,
                                                            use_mass_center=True),
                                            git_cfg=coco_ins_cfgs)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomChoice',
        transforms=[[dict(type='RandomChoiceResize', scales=[(base_img_size, base_img_size)], keep_ratio=False)],

                    [dict(type='RandomChoiceResize', scales=[(400, 4200), (500, 4200), (600, 4200)], keep_ratio=True),
                     dict(type='RandomCrop', crop_type='absolute_range', crop_size=(384, 600), allow_negative_crop=True),
                     dict(type='RandomChoiceResize', scales=[(base_img_size, base_img_size)], keep_ratio=False)]]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'task_name', 'head_cfg', 'git_cfg')),
    dict(type='Mask2DenseContour', restore_all=True,)]

coco_ins_test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, backend_args=backend_args),
    dict(type='Resize', scale=(base_img_size, base_img_size), keep_ratio=False),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='instance_segmentation', 
                                            head_cfg=dict(num_classes=num_classes,
                                                            num_vocal=num_vocal,
                                                            num_bins=num_bins,
                                                            dec_length=1+4+2+ray_num,
                                                            arg_max_inference=True,
                                                            ray_num=ray_num,
                                                            use_mass_center=True),
                                            git_cfg=coco_ins_cfgs)),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'task_name', 'head_cfg', 'git_cfg')),]

BDD100K_ins_train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='instance_segmentation', 
                                            head_cfg=dict(num_classes=8,
                                                            num_vocal=8+1+global_bin+1,
                                                            num_bins=global_bin,
                                                            dec_length=1+4+2+ray_num,
                                                            arg_max_inference=True,
                                                            ray_num=ray_num,
                                                            use_mass_center=True),
                                            git_cfg=BDD100K_ins_cfgs)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomChoice',
        transforms=[[dict(type='RandomChoiceResize', scales=[(base_img_size, base_img_size)], keep_ratio=False)],

                    [dict(type='RandomChoiceResize', scales=[(400, 4200), (500, 4200), (600, 4200)], keep_ratio=True),
                     dict(type='RandomCrop', crop_type='absolute_range', crop_size=(384, 600), allow_negative_crop=True),
                     dict(type='RandomChoiceResize', scales=[(base_img_size, base_img_size)], keep_ratio=False)]]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'task_name', 'head_cfg', 'git_cfg')),
    dict(type='Mask2DenseContour', restore_all=True,)]
## pipeline for semantic segmentation coco_stuff NOTE 164k images
coco_stuff164k_semseg_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='SegLoadAnnotations'),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='semantic_segmentation', 
                                            head_cfg=dict(num_classes=171,
                                                            num_vocal=172,
                                                            dec_length=16,
                                                            dec_pixel_resolution=[4, 4],
                                                            arg_max_inference=True,
                                                            ignore_index=255),
                                            git_cfg=coco_stuff164k_semseg_cfgs)),
    dict(type='RandomChoice',
        transforms=[[dict(type='RandomChoiceResize', scales=[(672, 672)], keep_ratio=False)],
                    [dict(type='RandomChoiceResize', scales=[(int(672 * x * 0.1), int(672 * x * 0.1))  for x in range(10, 21)], keep_ratio=False),
                     dict(type='SegRandomCrop', crop_size=(672, 672), cat_max_ratio=0.75),]]),
    dict(type='MMCVRandomFlip', prob=0.5),
    dict(type='SegPhotoMetricDistortion'),
    dict(type='PackSegInputs', meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 
                                          'reduce_zero_label', 'task_name', 'head_cfg', 'git_cfg'))]

nuimage_seg_semseg_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='SegLoadAnnotations',reduce_zero_label=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='semantic_segmentation', 
                                            head_cfg=dict(num_classes=31,
                                                            num_vocal=32,
                                                            dec_length=16,
                                                            dec_pixel_resolution=[4, 4],
                                                            arg_max_inference=True,
                                                            ignore_index=255),
                                            git_cfg=nuimage_seg_semseg_cfgs)),
    dict(type='RandomChoice',
        transforms=[[dict(type='RandomChoiceResize', scales=[(672, 672)], keep_ratio=False)],
                    [dict(type='RandomChoiceResize', scales=[(int(672 * x * 0.1), int(672 * x * 0.1))  for x in range(10, 21)], keep_ratio=False),
                     dict(type='SegRandomCrop', crop_size=(672, 672), cat_max_ratio=0.75),]]),
    dict(type='MMCVRandomFlip', prob=0.5),
    dict(type='SegPhotoMetricDistortion'),
    dict(type='PackSegInputs', meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 
                                          'reduce_zero_label', 'task_name', 'head_cfg', 'git_cfg'))]

BDD100K_semseg_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='SegLoadAnnotations'),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='semantic_segmentation', 
                                            head_cfg=dict(num_classes=19,
                                                            num_vocal=20,
                                                            dec_length=16,
                                                            dec_pixel_resolution=[4, 4],
                                                            arg_max_inference=True,
                                                            ignore_index=255),
                                            git_cfg=BDD100K_semseg_cfgs)),
    dict(type='RandomChoice',
        transforms=[[dict(type='RandomChoiceResize', scales=[(672, 672)], keep_ratio=False)],
                    [dict(type='RandomChoiceResize', scales=[(int(672 * x * 0.1), int(672 * x * 0.1))  for x in range(10, 21)], keep_ratio=False),
                     dict(type='SegRandomCrop', crop_size=(672, 672), cat_max_ratio=0.75),]]),
    dict(type='MMCVRandomFlip', prob=0.5),
    dict(type='SegPhotoMetricDistortion'),
    dict(type='PackSegInputs', meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 
                                          'reduce_zero_label', 'task_name', 'head_cfg', 'git_cfg'))]

mapillary_v2_semseg_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='SegLoadAnnotations'),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='semantic_segmentation', 
                                            head_cfg=dict(num_classes=124,
                                                            num_vocal=125,
                                                            dec_length=16,
                                                            dec_pixel_resolution=[4, 4],
                                                            arg_max_inference=True,
                                                            ignore_index=255),
                                            git_cfg=mapillary_v2_semseg_cfgs)),
    dict(type='RandomChoice',
        transforms=[[dict(type='RandomChoiceResize', scales=[(672, 672)], keep_ratio=False)],
                    [dict(type='RandomChoiceResize', scales=[(int(672 * x * 0.1), int(672 * x * 0.1))  for x in range(10, 21)], keep_ratio=False),
                     dict(type='SegRandomCrop', crop_size=(672, 672), cat_max_ratio=0.75),]]),
    dict(type='MMCVRandomFlip', prob=0.5),
    dict(type='SegPhotoMetricDistortion'),
    dict(type='PackSegInputs', meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 
                                          'reduce_zero_label', 'task_name', 'head_cfg', 'git_cfg'))]

pascal_context59_semseg_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='SegLoadAnnotations'),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='semantic_segmentation', 
                                            head_cfg=dict(num_classes=59,
                                                            num_vocal=60,
                                                            dec_length=16,
                                                            dec_pixel_resolution=[4, 4],
                                                            arg_max_inference=True,
                                                            ignore_index=255),
                                            git_cfg=pascal_context59_semseg_cfgs)),
    dict(type='RandomChoice',
        transforms=[[dict(type='RandomChoiceResize', scales=[(672, 672)], keep_ratio=False)],
                    [dict(type='RandomChoiceResize', scales=[(int(672 * x * 0.1), int(672 * x * 0.1))  for x in range(10, 21)], keep_ratio=False),
                     dict(type='SegRandomCrop', crop_size=(672, 672), cat_max_ratio=0.75),]]),
    dict(type='MMCVRandomFlip', prob=0.5),
    dict(type='SegPhotoMetricDistortion'),
    dict(type='PackSegInputs', meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 
                                          'reduce_zero_label', 'task_name', 'head_cfg', 'git_cfg'))]

ade20k_semseg_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='SegLoadAnnotations', reduce_zero_label=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='semantic_segmentation', 
                                            head_cfg=dict(num_classes=150,
                                                            num_vocal=151,
                                                            dec_length=16,
                                                            dec_pixel_resolution=[4, 4],
                                                            arg_max_inference=True,
                                                            ignore_index=255),
                                            git_cfg=ade20k_semseg_cfgs)),
    dict(type='RandomChoice',
        transforms=[[dict(type='RandomChoiceResize', scales=[(672, 672)], keep_ratio=False)],
                    [dict(type='RandomChoiceResize', scales=[(int(672 * x * 0.1), int(672 * x * 0.1))  for x in range(10, 21)], keep_ratio=False),
                     dict(type='SegRandomCrop', crop_size=(672, 672), cat_max_ratio=0.75),]]),
    dict(type='MMCVRandomFlip', prob=0.5),
    dict(type='SegPhotoMetricDistortion'),
    dict(type='PackSegInputs', meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 
                                          'reduce_zero_label', 'task_name', 'head_cfg', 'git_cfg'))]
ade20k_semseg_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(672, 672), keep_ratio=False),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='SegLoadAnnotations', reduce_zero_label=True),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='semantic_segmentation', 
                                            head_cfg=dict(num_classes=150,
                                                            num_vocal=151,
                                                            dec_length=16,
                                                            dec_pixel_resolution=[4, 4],
                                                            arg_max_inference=True,
                                                            ignore_index=255),
                                            git_cfg=ade20k_semseg_cfgs)),
    dict(type='PackSegInputs', meta_keys=('img_path', 'seg_map_path', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 
                                          'reduce_zero_label', 'task_name', 'head_cfg', 'git_cfg'))]
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
                                                            arg_max_inference=True,
                                                            ignore_index=-100,
                                                            beam_num=2),
                                            git_cfg=caption_cfgs)),
    dict(type='Resize', scale=(224, 224), interpolation='bicubic', backend='pillow'),
    dict(type='PackInputs', meta_keys=['image_id', 'img_shape', 'task_name', 'head_cfg', 'git_cfg']),
]

# pipeline for grounding
grounding_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='grounding', 
                            head_cfg=dict(num_classes=448+1,
                                            num_vocal=448+1,
                                            num_bins=448,
                                            dec_length=4,
                                            arg_max_inference=True),
                            git_cfg=grounding_cfgs)),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1,
                backend='cv2')
        ],
        prob=0.5),
    dict(
        type='mmdet.RandomCrop',
        crop_type='relative_range',
        crop_size=(0.8, 0.8),
        allow_negative_crop=False),
    dict(
        type='RandomChoiceResize',
        scales=[(224, 224)],
        keep_ratio=False),
    dict(type='CleanCaption', keys='text'),
    dict(
        type='PackInputs',
        algorithm_keys=['text', 'gt_bboxes',],
        meta_keys=['image_id','img_shape', 'scale_factor','task_name', 'head_cfg', 'git_cfg'],
    ),
]

grounding_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='AddMetaInfo', meta_dict=dict(task_name='grounding', 
                            head_cfg=dict(num_classes=448+1,
                                            num_vocal=448+1,
                                            num_bins=448,
                                            dec_length=4,
                                            arg_max_inference=True),
                            git_cfg=grounding_cfgs)),
    dict(
        type='Resize',
        scale=(224, 224),
        interpolation='bicubic',
        backend='pillow'),
    dict(type='CleanCaption', keys='text'),
    dict(
        type='PackInputs',
        algorithm_keys=['text', 'gt_bboxes', ],
        meta_keys=['image_id','img_shape','scale_factor','task_name', 'head_cfg', 'git_cfg'],
    ),
]

extra_ratio = 1.0
# down_ratio = 1 - extra_ratio
det_ratio=2./10. * extra_ratio
ins_ratio=2./10. * extra_ratio
seg_ratio=2./10.* extra_ratio
cap_ratio=2./10. * extra_ratio
ground_ratio=2./10 * extra_ratio

# detection: object365, openimage, lvisv1, VOC0712, nuimages, coco
# 1700, 1700,120,32,16,93,164
# 1/3*1700/3520,1/3*1700/3520,1/3*120/3520,0,1/3*16/180,1/3,1/3*164/180
# instance segmentation: lvis, openimage, nuimage, coco
# 120, 940, 93, 164
# 1/3*120/1060,1/3*940/1060,1/3,1/3
# semantic segmentation coco_stuff164k, nuimage_seg, Pascal Context, BDD100K, Mapillary V2, ade20k
# 164, 93, 10, 10, 25, 20
# 1/3*164/174,1/3*93/128,1/3*10/174,1/3*10/128,1/3*25/128,1/3
# caption: CCS(CC3M+CC12M+SBU), Visual Genome, COCO Caption
# 10000, 770, 164
# 1/2*10000/10770,1/2*770/10770,1/2
# grounding cfg: refcoco, refcoco+, refcocog, refclef, flickr30k
# 20, 20, 25, 20, 30
# 1/6,1/6,1/6,1/4,1/4

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='GroupMultiSourceNonMixedSampler', batch_size=1, 
                source_ratio=[
                    det_ratio*1/3*1700/3520, det_ratio*1/3*1700/3520, det_ratio*1/3*120/3520, det_ratio*1/3*16/180, det_ratio*1/6, det_ratio*1/3*164/180, det_ratio*1/6,
                    ins_ratio*1/3*120/1060, ins_ratio*1/3*940/1060, ins_ratio*2/9, ins_ratio*1/3., ins_ratio*1/9,
                    seg_ratio*1/3*164/174, seg_ratio*1/3*93/128, seg_ratio*1/3*10/174, seg_ratio*1/3*10/128, seg_ratio*1/3*25/128, seg_ratio*1/3,
                    cap_ratio*1/2*10000/10770, cap_ratio*1/2*770/10770, cap_ratio*1/2,
                    ground_ratio*1/5, ground_ratio*1/5, ground_ratio*1/5, ground_ratio*1/5, ground_ratio*1/5
                ], 
                if_group=[
                    True, True, True, True, True, True, True,# detection
                    True, True, True, True, True,# instance segmentation
                    False, False, False, False, False, False,# semantic segmentation
                    False, False, False, # caption
                    False, False, False, False, False # grounding
                ], shuffle=True),
    batch_sampler=None,
    dataset=dict(type='ConcatDataset',
                 ignore_keys=['reduce_zero_label', 'label_map', 'classes', 'palette', 'RELATION_MATRIX', 'dataset_type'],
        datasets=[
            # detection: object365, openimage, lvisv1,  VOC0712, nuimages, coco
            dict(type='Objects365V2Dataset',
                data_root='data/Objects365/Obj365_v2/',
                ann_file='annotations/zhiyuan_objv2_train.json',
                data_prefix=dict(img='train/'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=obj365_det_train_pipeline,
                backend_args=backend_args),
            dict(type='ClassBalancedDataset',
                oversample_thr=1./601,
                dataset=dict(type='OpenImagesDataset',
                data_root='data/OpenImages/',
                ann_file='annotations/oidv6-train-annotations-bbox.csv',
                data_prefix=dict(img='OpenImages/train/'),
                label_file='annotations/class-descriptions-boxable.csv',
                hierarchy_file='annotations/bbox_labels_600_hierarchy.json',
                meta_file='annotations/train-image-metas.pkl',
                pipeline=openimages_det_train_pipeline,
                backend_args=backend_args)),
            dict(type='ClassBalancedDataset',
                oversample_thr=1e-3,
                dataset=dict(type='LVISV1Dataset',
                data_root='data/lvis_v1/',
                ann_file='annotations/lvis_v1_train.json',
                data_prefix=dict(img=''),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=lvisv1_det_train_pipeline,
                backend_args=backend_args)),
            dict(
                type='RepeatDataset',
                times=3,
                dataset=dict(
                type='ConcatDataset',
                # VOCDataset will add different `dataset_type` in dataset.metainfo,
                # which will get error if using ConcatDataset. Adding
                # `ignore_keys` can avoid this error.
                ignore_keys=['dataset_type'],
                datasets=[
                    dict(type='VOCDataset',
                        data_root='data/VOCdevkit/',
                        ann_file='VOC2007/ImageSets/Main/trainval.txt',
                        data_prefix=dict(sub_data_root='VOC2007/'),
                        filter_cfg=dict(
                            filter_empty_gt=True, min_size=32, bbox_min_size=32),
                        pipeline=voc0712_det_train_pipeline,
                        backend_args=backend_args),
                    dict(type='VOCDataset',
                        data_root='data/VOCdevkit/',
                        ann_file='VOC2012/ImageSets/Main/trainval.txt',
                        data_prefix=dict(sub_data_root='VOC2012/'),
                        filter_cfg=dict(
                            filter_empty_gt=True, min_size=32, bbox_min_size=32),
                        pipeline=voc0712_det_train_pipeline,
                        backend_args=backend_args),
            ])),
            dict(
                type='NuimageDataset',
                ann_file='data/nuimages/' + 'annotations/nuimages_v1.0-train.json',
                data_prefix=dict(img='data/nuimages/'),
                pipeline=nuimage_det_train_pipeline),
            dict(type='CocoDataset',
                data_root='data/coco/',
                ann_file='annotations/instances_train2017.json',
                data_prefix=dict(img='train2017/'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=coco_det_train_pipeline,
                backend_args=backend_args),
            dict(type='BDD100KDetDataset',
                data_root='data/bdd100k_det/',
                ann_file='labels_coco/det_train_coco.json',
                data_prefix=dict(img='train/'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=BDD100K_det_train_pipeline,
                backend_args=backend_args),
            # instance segmentation: lvis, openimage, nuimage, coco
            dict(type='ClassBalancedDataset',
                oversample_thr=1e-3,
                dataset=dict(type='LVISV1Dataset',
                data_root='data/lvis_v1/',
                ann_file='annotations/lvis_v1_train.json',
                data_prefix=dict(img=''),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=lvisv1_ins_train_pipeline,
                backend_args=backend_args)),
            dict(type='ClassBalancedDataset',
                oversample_thr=1./601,
                dataset=dict(type='OpenImagesDatasetInseg',
                data_root='data/OpenImages/',
                ann_file='annotations/train-annotations-object-segmentation_sort_resize.csv',
                data_prefix=dict(img='OpenImages/train/',seg='segmentation/train'),
                label_file='annotations/class-descriptions-boxable.csv',
                hierarchy_file='annotations/bbox_labels_600_hierarchy.json',
                meta_file='annotations/train-image-metas-dict.pkl',
                pipeline=openimage_ins_train_pipeline,
                backend_args=backend_args)),
            dict(
                type='NuimageDataset',
                ann_file='data/nuimages/' + 'annotations/nuimages_v1.0-train.json',
                data_prefix=dict(img='data/nuimages/'),
                pipeline=nuimage_ins_train_pipeline),
            dict(type='CocoDataset',
                data_root='data/coco/',
                ann_file='annotations/instances_train2017.json',
                data_prefix=dict(img='train2017/'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=coco_ins_train_pipeline,
                backend_args=backend_args),
            dict(type='BDD100KInsDataset',
                data_root='data/bdd100k_ins/',
                ann_file='annotations/bdd_ins_train_coco.json',
                data_prefix=dict(img='train/'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=BDD100K_ins_train_pipeline,
                backend_args=backend_args),
            # semantic segmentation coco_stuff164k, nuimage_seg, Pascal Context, BDD100K, Mapillary V2,  ade20k
            dict(type='COCOStuffDataset',
                data_root='data/coco_stuff164k',
                data_prefix=dict(
                    img_path='images/train2017', seg_map_path='annotations/train2017'),
                pipeline=coco_stuff164k_semseg_train_pipeline),
            dict(type='NuimageSegDataset',
                data_root='data/nuimages_seg',
                data_prefix=dict(
                    img_path='images/training', seg_map_path='annotations/training'),
                pipeline=nuimage_seg_semseg_train_pipeline),
            dict(type='PascalContextDataset59',
                data_root='data/VOCdevkit/VOC2010/',
                data_prefix=dict(
                    img_path='JPEGImages', seg_map_path='SegmentationClassContext'),
                ann_file='ImageSets/SegmentationContext/train.txt',
                pipeline=pascal_context59_semseg_train_pipeline),
            dict(type='BDD100KDataset',
                data_root='data/bdd100k/',
                data_prefix=dict(
                    img_path='images/10k/train',
                    seg_map_path='labels/sem_seg/masks/train'),
                pipeline=BDD100K_semseg_train_pipeline),
            dict(type='MapillaryDataset_v2',
                data_root='data/mapillary/',
                data_prefix=dict(
                    img_path='training/images', seg_map_path='training/v2.0/labels'),
                pipeline=mapillary_v2_semseg_train_pipeline),
            dict(type='ADE20KDataset',
                data_root='data/ade/ADEChallengeData2016',
                data_prefix=dict(img_path='images/training', seg_map_path='annotations/training'),
                pipeline=ade20k_semseg_train_pipeline),
            # caption: CCS, VG, COCO Caption
            dict(type='CCSCaption',
                data_root='data/ccs',
                ann_file='ccs_synthetic_filtered_mask_12m.json',
                pipeline=caption_train_pipeline),
            dict(type='VGCaption',
                data_root='data/vg',
                ann_file='vg.json',
                pipeline=caption_train_pipeline),
            dict(type='COCOCaption',
                data_root='data/coco_2014',
                ann_file='annotations/coco_karpathy_train.json',
                pipeline=caption_train_pipeline),
            # grounding: refcoco, refcoco+, refcocog
            dict(type='RefCOCO',
                data_root='data/coco_2014',
                data_prefix='train2014',
                ann_file='refcoco/instances.json',
                split_file='refcoco/refs(unc).p',
                split='train',
                pipeline=grounding_train_pipeline),
            dict(
                type='RefCOCO',
                data_root='data/coco_2014',
                data_prefix='train2014',
                ann_file='refcoco+/instances.json',
                split_file='refcoco+/refs(unc).p',
                split='train',
                pipeline=grounding_train_pipeline),
            dict(
                type='RefCOCO',
                data_root='data/coco_2014',
                data_prefix='train2014',
                ann_file='refcocog/instances.json',
                split_file='refcocog/refs(umd).p',
                split='train',
                pipeline=grounding_train_pipeline),
            dict(type='RefCOCO',
                data_root='data/refclef',
                data_prefix='saiapr_tc-12',
                ann_file='instances.json',
                split_file='refs(unc).p',
                split='train',
                pipeline=grounding_train_pipeline),
            dict(type='Flickr30K',
                data_root='data/flickr30k',
                data_prefix='flickr30k-images',
                ann_file='flickr_train.pth',
                split='train',
                pipeline=grounding_train_pipeline),
            ]),       
    )

test_pipeline = coco_ins_test_pipeline
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='data/coco/',
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=coco_ins_test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

# construct extra val dataloaders
extra_val_dataloaders = [
    dict(batch_size=1,
        num_workers=1,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='CocoDataset',
            data_root='data/coco/',
            ann_file='annotations/instances_val2017.json',
            data_prefix=dict(img='val2017/'),
            test_mode=True,
            pipeline=coco_det_test_pipeline,
            backend_args=None)),
    dict(batch_size=2,
        num_workers=1,
        persistent_workers=True,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='COCOCaption',
            data_root='data/coco_2014',
            ann_file='annotations/coco_karpathy_test.json',
            pipeline=caption_test_pipeline)),
    dict(batch_size=1, # batch inference is not supported now.
        num_workers=1,
        persistent_workers=True,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='ADE20KDataset',
            data_root='data/ade/ADEChallengeData2016',
            data_prefix=dict(
                img_path='images/validation',
                seg_map_path='annotations/validation'),
            pipeline=ade20k_semseg_test_pipeline)),
    dict(batch_size=1,
        num_workers=1,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='RefCOCO',
            data_root='data/coco_2014',
            data_prefix='train2014',
            ann_file='refcoco/instances.json',
            split_file='refcoco/refs(unc).p',
            split='val',  # or 'testB'
            pipeline=grounding_test_pipeline))
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    # optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.05),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'backbone.layers.0': dict(lr_mult=0.1),
            'backbone.layers.1': dict(lr_mult=0.1),
            'backbone.layers.2': dict(lr_mult=0.1),
            'backbone.layers.3': dict(lr_mult=0.1),
            'backbone.layers.4': dict(lr_mult=0.1),
            'backbone.layers.5': dict(lr_mult=0.1),
            'backbone.layers.6': dict(lr_mult=0.1),
            'backbone.layers.7': dict(lr_mult=0.1),
            'backbone.layers.8': dict(lr_mult=0.1),
            'backbone.layers.9': dict(lr_mult=0.1),
            'backbone.layers.10': dict(lr_mult=0.1),
            'backbone.layers.11': dict(lr_mult=0.1),
            'backbone.layers.12': dict(lr_mult=0.1),
            'backbone.layers.13': dict(lr_mult=0.1),
            'backbone.layers.14': dict(lr_mult=0.1),
            'backbone.layers.15': dict(lr_mult=0.1),
            'backbone.layers.16': dict(lr_mult=0.1),
            'backbone.layers.17': dict(lr_mult=0.15625),
            'backbone.layers.18': dict(lr_mult=0.2125),
            'backbone.layers.19': dict(lr_mult=0.26875),
            'backbone.layers.20': dict(lr_mult=0.325),
            'backbone.layers.21': dict(lr_mult=0.38125),
            'backbone.layers.22': dict(lr_mult=0.4375),
            'backbone.layers.23': dict(lr_mult=0.49375),
            'backbone.layers.24': dict(lr_mult=0.55),
            'backbone.layers.25': dict(lr_mult=0.60625),
            'backbone.layers.26': dict(lr_mult=0.6625),
            'backbone.layers.27': dict(lr_mult=0.71875),
            'backbone.layers.28': dict(lr_mult=0.7750),
            'backbone.layers.29': dict(lr_mult=0.83125),
            'backbone.layers.30': dict(lr_mult=0.8875),
            'backbone.layers.31': dict(lr_mult=0.94375),
            'backbone.layers.32': dict(lr_mult=1.0),
            'backbone.layers.33': dict(lr_mult=1.0),
            'backbone.layers.34': dict(lr_mult=1.0),
            'backbone.layers.35': dict(lr_mult=1.0),
            'backbone.layers.36': dict(lr_mult=1.0),
            'backbone.layers.37': dict(lr_mult=1.0),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))

val_evaluator = dict(
    ann_file='data/coco/annotations/instances_val2017.json',
    backend_args=None,
    format_only=False,
    metric=[
        'bbox',
        'segm',
    ],
    type='CocoMetric')
test_evaluator = val_evaluator
extra_val_evaluators = [
    dict(type='CocoMetric',
        ann_file='data/coco/' + 'annotations/instances_val2017.json',
        metric='bbox',
        format_only=False,
        backend_args=backend_args),
    dict(type='COCOCaption',
        ann_file='data/coco_2014/annotations/coco_karpathy_test_gt.json',),
    dict(type='IoUMetric', iou_metrics=['mIoU']),
    dict(type='VisualGroundingMetric'),
]

# learning policy
max_iters=480000
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=max_iters, val_interval=10000)
val_cfg = dict(type='MultiSourceValLoop', extra_dataloaders=extra_val_dataloaders, extra_evaluators=extra_val_evaluators)
# test_cfg = dict(type='TestLoop')
# val_cfg = dict(type='ValLoop')
test_cfg = val_cfg

# learning policy
param_scheduler = [
    # dict(type='LinearLR',
    #     start_factor=0.1,
    #     begin=0,
    #     end=5000,
    #     by_epoch=False),
    dict(
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
auto_scale_lr = dict(enable=True, base_batch_size=24)


# find_unused_parameters = True

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=500, max_keep_ckpts=50),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

log_processor = dict(type='LogProcessor', window_size=4000, by_epoch=False)
