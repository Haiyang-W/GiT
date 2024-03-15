# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from typing import List

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class BDD100KInsDataset(CocoDataset):
    """Kitti dataset for detection."""

    METAINFO = {
        'classes':
        ('pedestrian', 'rider', 'car','truck','bus','train','motorcycle','bicycle'),
        'palette': None
        # ([220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
        # [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32],[250, 170,30], [220, 220, 0])
    }
