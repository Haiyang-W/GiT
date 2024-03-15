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
class PotsdamDataset(BaseSegDataset):
    """ISPRS Potsdam dataset.

    In segmentation map annotation for Potsdam dataset, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    METAINFO = dict(
        classes=('impervious_surface', 'building', 'low_vegetation', 'tree',
                 'car', 'clutter'),
        palette=[[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
                 [255, 255, 0], [255, 0, 0]])

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
            # 5-shot
            names = ['6_9_0_1536_512_2048', '6_12_5488_3584_6000_4096', '7_10_3072_3072_3584_3584', '2_12_1024_2048_1536_2560', '4_10_2048_4096_2560_4608', '5_12_512_4608_1024_5120', '4_12_5488_5120_6000_5632', '4_10_512_5120_1024_5632', '6_7_4096_512_4608_1024', '6_9_3072_5120_3584_5632', '3_11_5488_1536_6000_2048', '2_12_4096_0_4608_512', '3_11_4096_2560_4608_3072', '7_7_2048_5120_2560_5632', '6_11_1536_0_2048_512', '2_12_0_3072_512_3584', '4_12_1024_4096_1536_4608', '7_11_5120_3584_5632_4096', '6_11_3072_0_3584_512', '4_10_5120_512_5632_1024', '4_10_5120_4096_5632_4608', '7_10_3584_3072_4096_3584', '7_12_2048_2560_2560_3072', '4_11_2560_0_3072_512', '7_10_2560_3072_3072_3584', '2_11_4608_1024_5120_1536', '6_12_2560_3072_3072_3584', '5_12_2560_5488_3072_6000', '6_7_5120_2560_5632_3072', '2_10_4608_4096_5120_4608']
            # 10-shot
            # names = ['6_9_0_1536_512_2048', '6_12_5488_3584_6000_4096', '7_10_3072_3072_3584_3584', '2_12_1024_2048_1536_2560', '4_10_2048_4096_2560_4608', '4_12_5488_5120_6000_5632', '4_10_512_5120_1024_5632', '6_9_3072_5120_3584_5632', '3_11_5488_1536_6000_2048', '3_11_4096_2560_4608_3072', '5_12_512_4608_1024_5120', '6_7_4096_512_4608_1024', '7_7_2048_5120_2560_5632', '6_11_1536_0_2048_512', '4_12_1024_4096_1536_4608', '7_11_5120_3584_5632_4096', '6_11_3072_0_3584_512', '4_10_5120_512_5632_1024', '6_12_2560_3072_3072_3584', '4_10_5120_4096_5632_4608', '2_12_4096_0_4608_512', '2_12_0_3072_512_3584', '2_11_4608_1024_5120_1536', '6_7_5120_2560_5632_3072', '2_10_4608_4096_5120_4608', '3_12_0_1024_512_1536', '7_10_3584_3072_4096_3584', '6_11_1536_1536_2048_2048', '7_12_2048_2560_2560_3072', '4_11_2048_0_2560_512', '4_11_2560_0_3072_512', '7_10_2560_3072_3072_3584', '6_7_3072_3072_3584_3584', '7_11_4096_5488_4608_6000', '5_12_5120_2560_5632_3072', '5_10_3072_3584_3584_4096', '4_10_3072_2048_3584_2560', '3_10_1536_3584_2048_4096', '7_7_4608_3072_5120_3584', '7_10_3584_4096_4096_4608', '4_11_3072_5488_3584_6000', '3_11_3584_1024_4096_1536', '7_8_4096_4608_4608_5120', '4_12_0_1536_512_2048', '7_7_5488_3584_6000_4096', '6_9_2560_512_3072_1024', '7_11_2048_5488_2560_6000', '5_10_5120_2560_5632_3072', '5_11_3584_1024_4096_1536', '5_12_2560_2560_3072_3072', '5_12_2560_5488_3072_6000', '3_11_2048_0_2560_512', '7_8_1024_1024_1536_1536', '5_10_1536_3584_2048_4096', '5_11_1024_5488_1536_6000', '7_10_4608_5488_5120_6000', '6_11_1024_4096_1536_4608', '7_8_4608_1536_5120_2048', '3_10_5120_4608_5632_5120', '4_10_3584_3584_4096_4096']
            for data_info in data_list:
                if osp.basename(data_info['img_path']).split('.')[0] in names:
                    select_list.append(data_info)
            assert len(select_list) > 0
            print("support sample",select_list)
            return select_list
        else:
            return data_list

