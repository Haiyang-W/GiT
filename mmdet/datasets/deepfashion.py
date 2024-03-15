# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import List, Union, Optional

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .coco import CocoDataset
from .api_wrappers import COCO


@DATASETS.register_module()
class DeepFashionDataset(CocoDataset):
    """Dataset for DeepFashion."""

    METAINFO = {
        'classes': ('top', 'skirt', 'leggings', 'dress', 'outer', 'pants',
                    'bag', 'neckwear', 'headwear', 'eyeglass', 'belt',
                    'footwear', 'hair', 'skin', 'face'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(0, 192, 64), (0, 64, 96), (128, 192, 192), (0, 64, 64),
                    (0, 192, 224), (0, 192, 192), (128, 192, 64), (0, 192, 96),
                    (128, 32, 192), (0, 0, 224), (0, 0, 64), (0, 160, 192),
                    (128, 0, 96), (128, 0, 192), (0, 32, 192)]
    }
    def __init__(self,
                 *args,
                 seg_map_suffix: str = '.png',
                 proposal_file: Optional[str] = None,
                 file_client_args: dict = None,
                 backend_args: dict = None,
                 support_num: int = -1,
                 **kwargs) -> None:
        self.seg_map_suffix = seg_map_suffix
        self.proposal_file = proposal_file
        self.backend_args = backend_args
        self.support_num = support_num
        if file_client_args is not None:
            raise RuntimeError(
                'The `file_client_args` is deprecated, '
                'please use `backend_args` instead, please refer to'
                'https://github.com/open-mmlab/mmdetection/blob/main/configs/_base_/datasets/coco_detection.py'  # noqa: E501
            )
        super().__init__(*args, **kwargs)
    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco
        if self.support_num != -1:
            print(data_list[:self.support_num])
            return data_list[:self.support_num]
        else:
            return data_list
