# Copyright (c) OpenMMLab. All rights reserved.
from ..registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class NuimageSegDataset(BaseSegDataset):
    """NuimageSeg dataset.

    In segmentation map annotation for NuimageSeg, 0 stands for background, which
    is not included in 31 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        classes=('animal','adult pedestrian','child pedestrian','construction worker',
        'portable personal mobility vehicle','police officer','stroller','wheelchair',
        'barrier','debris','pushable pullable objects','traffic cone','bicycle rack',
        'bicycle','bendy bus','rigid bus','car','construction vehicle','ambulance',
        'police vehicle','motorcycle','trailer','truck','driveable_surface','sidewalk',
        'terrain','flat.other','static.manmade','static.vegetation','static.other',
        'ego vehicle'),
        palette=[[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                 [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                 [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                 [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                 [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                 [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                 [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                 [255, 9, 92], [112, 9, 255], [8, 255, 214]])

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
