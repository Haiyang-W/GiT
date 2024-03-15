# Copyright (c) OpenMMLab. All rights reserved.
import os.path
from typing import Optional
import copy
import os.path as osp
from typing import List, Union
import mmengine
from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .coco import CocoDataset
from .api_wrappers import COCO



@DATASETS.register_module()
class V3DetDataset(CocoDataset):
    """Dataset for V3Det."""

    METAINFO = {
        'classes': None,
        'palette': None,
    }

    def __init__(
            self,
            *args,
            metainfo: Optional[dict] = None,
            data_root: str = '',
            label_file='annotations/category_name_13204_v3det_2023_v1.txt',  # noqa
            **kwargs) -> None:
        class_names = tuple(
            mmengine.list_from_file(os.path.join(data_root, label_file)))
        if metainfo is None:
            metainfo = {'classes': class_names}
        super().__init__(
            *args, data_root=data_root, metainfo=metainfo, **kwargs)
    
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
        return data_list