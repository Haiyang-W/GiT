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
class DRIVEDataset(BaseSegDataset):
    """DRIVE dataset.

    In segmentation map annotation for DRIVE, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '_manual1.png'.
    """
    METAINFO = dict(
        classes=('background', 'vessel'),
        palette=[[120, 120, 120], [6, 230, 230]]
        # classes=('vessel',),
        # palette=[[6, 230, 230]]
        )

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='_manual1.png',
                 reduce_zero_label=False,
                 support_num=-1,
                  **kwargs) -> None:
        self.support_num = support_num
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
        assert fileio.exists(
            self.data_prefix['img_path'], backend_args=self.backend_args)
    
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
            print("support sample",data_list[:self.support_num])
            return data_list[:self.support_num]
        else:
            return data_list

