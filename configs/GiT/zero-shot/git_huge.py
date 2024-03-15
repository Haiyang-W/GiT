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
