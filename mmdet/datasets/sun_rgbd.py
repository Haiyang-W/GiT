# Copyright (c) OpenMMLab. All rights reserved.
from ..registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class SunRGBDDataset(BaseSegDataset):
    """NuimageSeg dataset.

    In segmentation map annotation for NuimageSeg, 0 stands for background, which
    is not included in 31 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        classes=('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 
        'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 
        'pillow', 'mirror', 'floor mat', 'clothes', 'ceiling', 'books', 'refridgerator', 
        'television', 'paper', 'towel', 'shower curtain', 'box', 'whiteboard', 'person', 
        'night stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag'),
        palette=[(120, 120, 120), (180, 120, 120), (6, 230, 230), (80, 50, 50),
               (4, 200, 3), (120, 120, 80), (140, 140, 140), (204, 5, 255),
               (230, 230, 230), (4, 250, 7), (224, 5, 255), (235, 255, 7),
               (150, 5, 61), (120, 120, 70), (8, 255, 51), (255, 6, 82),
               (143, 255, 140), (204, 255, 4), (255, 51, 7), (204, 70, 3),
               (0, 102, 200), (61, 230, 250), (255, 6, 51), (11, 102, 255),
               (255, 7, 71), (255, 9, 224), (9, 7, 230), (220, 220, 220),
               (255, 9, 92), (112, 9, 255), (8, 255, 214), (7, 255, 224),
               (255, 184, 6), (10, 255, 71), (255, 41, 10), (7, 255, 255),
               (224, 255, 8),])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
