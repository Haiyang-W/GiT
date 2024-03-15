# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List

import mmengine
import numpy as np
from mmengine.dataset import BaseDataset
from pycocotools.coco import COCO
import torch

from mmdet.registry import DATASETS


@DATASETS.register_module()
class Flickr30K(BaseDataset):
    """Flickr dataset.

    RefCOCO is a popular dataset used for the task of visual grounding.
    Here are the steps for accessing and utilizing the
    RefCOCO dataset.

    You can access the RefCOCO dataset from the official source:
    https://github.com/lichengunc/refer

    The RefCOCO dataset is organized in a structured format: ::

        FeaturesDict({
            'coco_annotations': Sequence({
                'area': int64,
                'bbox': BBoxFeature(shape=(4,), dtype=float32),
                'id': int64,
                'label': int64,
            }),
            'image': Image(shape=(None, None, 3), dtype=uint8),
            'image/id': int64,
            'objects': Sequence({
                'area': int64,
                'bbox': BBoxFeature(shape=(4,), dtype=float32),
                'gt_box_index': int64,
                'id': int64,
                'label': int64,
                'refexp': Sequence({
                    'raw': Text(shape=(), dtype=string),
                    'refexp_id': int64,
                }),
            }),
        })

    Args:
        ann_file (str): Annotation file path.
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        data_prefix (str): Prefix for training data.
        pipeline (Sequence): Processing pipeline. Defaults to an empty tuple.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """

    def __init__(self,
                 data_root,
                 ann_file,
                 data_prefix,
                 split='train',
                 **kwargs):
        self.split = split

        super().__init__(
            data_root=data_root,
            data_prefix=dict(img_path=data_prefix),
            ann_file=ann_file,
            **kwargs,
        )

    def load_data_list(self) -> List[dict]:
        """Load data list."""
        data_list = []
        annotations = torch.load(self.ann_file)
        img_prefix = self.data_prefix['img_path']
        join_path = mmengine.fileio.get_file_backend(img_prefix).join_path
        for refer in annotations:
            file_name, bbox, text = refer
            image_id = file_name.split('.')[0]
            data_info = {
                    'img_path': join_path(img_prefix, file_name),
                    'image_id': image_id,
                    'text': text,
                    'gt_bboxes': bbox[None, :],
                }
            data_list.append(data_info)

        if len(data_list) == 0:
            raise ValueError(f'No sample in split "{self.split}".')

        return data_list
