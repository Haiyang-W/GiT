# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from typing import List

import mmengine
from mmengine.dataset import BaseDataset
from mmengine.fileio import get_file_backend

from mmdet.registry import DATASETS
import os

@DATASETS.register_module()
class VGCaption(BaseDataset):
    """Visual Genome Caption dataset.

    Args:
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``..
        ann_file (str): Annotation file path.
        data_prefix (dict): Prefix for data field. Defaults to
            ``dict(img_path='')``.
        pipeline (Sequence): Processing pipeline. Defaults to an empty tuple.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """

    def load_data_list(self) -> List[dict]:
        """Load data list."""
        img_prefix = self.data_prefix['img_path']
        annotations = mmengine.load(self.ann_file)
        file_backend = get_file_backend(img_prefix)

        data_list = []
        for i,ann in enumerate(annotations):
            data_info = {
                'image_id': Path(ann['image']).stem.split('_')[-1],
                'img_path': file_backend.join_path(img_prefix, ann['image']),
                'gt_caption': ann['caption'],
            }
            data_list.append(data_info)
        # 'image': '/export/share/datasets/vision/visual-genome/image/1.jpg', 'caption': 'trees line the sidewalk', 'image_id': 'vg_1'}

        return data_list
