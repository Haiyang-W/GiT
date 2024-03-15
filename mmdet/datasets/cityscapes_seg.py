# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class CityscapesSegDataset(BaseSegDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    METAINFO = dict(
        classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                 'motorcycle', 'bicycle'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170,
                                                    30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])
    # METAINFO = dict(
    #     classes=(
    #         'ego vehicle', 'rectification border', 'out of roi', 'static','dynamic','ground','road','sidewalk','parking','rail track','building','wall' ,
    #         'fence' ,'guard rail' ,'bridge' ,'tunnel','pole'   ,'polegroup','traffic light' ,'traffic sign' ,'vegetation'   ,'terrain','sky' ,
    #         'person' ,'rider','car','truck' ,'bus'  ,'caravan','trailer' ,'train'  ,'motorcycle','bicycle',
    #     ),
    #      palette=[[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
    #              [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
    #              [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
    #              [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
    #              [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
    #              [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
    #              [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
    #              [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
    #              [255, 184, 6]]
    # )

    def __init__(self,
                 img_suffix='_leftImg8bit.png',
                 seg_map_suffix='_gtFine_labelTrainIds.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
