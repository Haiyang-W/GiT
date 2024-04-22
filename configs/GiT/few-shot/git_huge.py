# just placeholder
ray_num = 24
global_bin = 2240
num_bins = global_bin
num_classes = 80
num_vocal = (num_bins + 1) + num_classes + 1 # use in tasks
pretrained = 'https://download.openmmlab.com/mmclassification/v1/vit_sam/vit-huge-p16_sam-pre_3rdparty_sa1b-1024px_20230411-3f13c653.pth'

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

test_cfg = dict(type='TestLoop')
val_cfg = dict(type='ValLoop')

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
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