# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import Callable, Dict, List, Optional, Sequence, Union

import mmengine
import mmengine.fileio as fileio
import numpy as np
from mmengine.dataset import BaseDataset, Compose

from mmdet.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class LoveDADataset(BaseSegDataset):
    """LoveDA dataset.

    In segmentation map annotation for LoveDA, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    METAINFO = dict(
        classes=('background', 'building', 'road', 'water', 'barren', 'forest',
                 'agricultural'),
        palette=[[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
                 [159, 129, 183], [0, 255, 0], [255, 195, 128]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 support_num=-1,
                 **kwargs) -> None:
        self.support_num = support_num
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        if osp.isfile(self.ann_file):
            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args)
            for line in lines:
                img_name = line.strip()
                data_info = dict(
                    img_path=osp.join(img_dir, img_name + self.img_suffix))
                if ann_dir is not None:
                    seg_map = img_name + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                if self.return_classes:
                    data_info['classes'] = self.metainfo['classes']
                data_list.append(data_info)
        else:
            for img in fileio.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=self.img_suffix,
                    recursive=True,
                    backend_args=self.backend_args):
                data_info = dict(img_path=osp.join(img_dir, img))
                if ann_dir is not None:
                    seg_map = img.replace(self.img_suffix, self.seg_map_suffix)
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                if self.return_classes:
                    data_info['classes'] = self.metainfo['classes']
                data_list.append(data_info)
            data_list = sorted(data_list, key=lambda x: x['img_path'])
        if self.support_num != -1:
            select_list = []
            # select fist k 
            names = ['2327', '1828', '27', '2475', '175',
             '2061', '1976', '712', '2422', '2268',
              '397', '934', '1923', '1769', '606', 
              '331', '553', '775', '997', '2363',
            '394', '1247', '1395', '772', '1135',
             '228', '1040', '172', '931', '1342',
              '1093', '225', '603', '2048', '550']
            for data_info in data_list:
                if osp.basename(data_info['img_path']).split('.')[0] in names:
                    select_list.append(data_info)
            assert len(select_list) > 0
            print("support sample",select_list)
            return select_list
        else:
            return data_list

