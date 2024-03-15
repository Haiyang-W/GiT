# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import itertools
import copy
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union
import tempfile
import os.path as osp

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from terminaltables import AsciiTable
from mmengine.fileio import dump, get_local_path, load

from mmdet.registry import METRICS
from ..functional import eval_map,eval_recalls
import torch
from mmdet.structures.mask import encode_mask_results
from mmdet.datasets.api_wrappers import COCO, COCOeval


@METRICS.register_module()
class OpenImagesInSegMetric(BaseMetric):
    """OpenImages evaluation metric.

    Evaluate detection mAP for OpenImages. Please refer to
    https://storage.googleapis.com/openimages/web/evaluation.html for more
    details.

    Args:
        iou_thrs (float or List[float]): IoU threshold. Defaults to 0.5.
        ioa_thrs (float or List[float]): IoA threshold. Defaults to 0.5.
        scale_ranges (List[tuple], optional): Scale ranges for evaluating
            mAP. If not specified, all bounding boxes would be included in
            evaluation. Defaults to None
        use_group_of (bool): Whether consider group of groud truth bboxes
            during evaluating. Defaults to True.
        get_supercategory (bool): Whether to get parent class of the
            current class. Default: True.
        filter_labels (bool): Whether filter unannotated classes.
            Default: True.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """
    default_prefix: Optional[str] = 'openimages'

    def __init__(self,
                 metric: Union[str, List[str]] = 'bbox',
                 classwise: bool = False,
                 iou_thrs: Union[float, List[float]] = 0.5,
                 ioa_thrs: Union[float, List[float]] = 0.5,
                 metric_items: Optional[Sequence[str]] = None,
                 scale_ranges: Optional[List[tuple]] = None,
                 use_group_of: bool = True,
                 get_supercategory: bool = True,
                 filter_labels: bool = True,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError(
                    "metric should be one of 'bbox', 'segm', 'proposal', "
                    f"'proposal_fast', but got {metric}.")
        self.iou_thrs = [iou_thrs] if isinstance(iou_thrs, float) else iou_thrs
        self.ioa_thrs = [ioa_thrs] if (isinstance(ioa_thrs, float)
                                       or ioa_thrs is None) else ioa_thrs
        assert isinstance(self.iou_thrs, list) and isinstance(
            self.ioa_thrs, list)
        assert len(self.iou_thrs) == len(self.ioa_thrs)

        self.scale_ranges = scale_ranges
        self.use_group_of = use_group_of
        self.get_supercategory = get_supercategory
        self.filter_labels = filter_labels
        self.outfile_prefix = outfile_prefix
        self.format_only = format_only
        self.metric_items = metric_items
        self.classwise = classwise
        # handle dataset lazy init
        self.cat_ids = None
        self.img_ids = None
        self._coco_api = None

    def _get_supercategory_ann(self, instances: List[dict]) -> List[dict]:
        """Get parent classes's annotation of the corresponding class.

        Args:
            instances (List[dict]): A list of annotations of the instances.

        Returns:
            List[dict]: Annotations extended with super-category.
        """
        supercat_instances = []
        relation_matrix = self.dataset_meta['RELATION_MATRIX']
        for instance in instances:
            labels = np.where(relation_matrix[instance['bbox_label']])[0]
            for label in labels:
                if label == instance['bbox_label']:
                    continue
                new_instance = copy.deepcopy(instance)
                new_instance['bbox_label'] = label
                supercat_instances.append(new_instance)
        return supercat_instances

    def _process_predictions(self, pred_bboxes: np.ndarray,
                             pred_scores: np.ndarray, pred_labels: np.ndarray, pred_masks: list,
                             gt_instances: list,
                             image_level_labels: np.ndarray) -> tuple:
        """Process results of the corresponding class of the detection bboxes.

        Note: It will choose to do the following two processing according to
        the parameters:

        1. Whether to add parent classes of the corresponding class of the
        detection bboxes.

        2. Whether to ignore the classes that unannotated on that image.

        Args:
            pred_bboxes (np.ndarray): bboxes predicted by the model
            pred_scores (np.ndarray): scores predicted by the model
            pred_labels (np.ndarray): labels predicted by the model
            gt_instances (list): ground truth annotations
            image_level_labels (np.ndarray): human-verified image level labels

        Returns:
            tuple: Processed bboxes, scores, and labels.
        """
        processed_bboxes = copy.deepcopy(pred_bboxes)
        processed_scores = copy.deepcopy(pred_scores)
        processed_labels = copy.deepcopy(pred_labels)
        processed_masks = copy.deepcopy(pred_masks)
        gt_labels = np.array([ins['bbox_label'] for ins in gt_instances],
                             dtype=np.int64)
        if image_level_labels is not None:
            allowed_classes = np.unique(
                np.append(gt_labels, image_level_labels))
        else:
            allowed_classes = np.unique(gt_labels)
        relation_matrix = self.dataset_meta['RELATION_MATRIX']
        pred_classes = np.unique(pred_labels)
        for pred_class in pred_classes:
            classes = np.where(relation_matrix[pred_class])[0]
            for cls in classes:
                if (cls in allowed_classes and cls != pred_class
                        and self.get_supercategory):
                    # add super-supercategory preds
                    index = np.where(pred_labels == pred_class)[0]
                    processed_scores = np.concatenate(
                        [processed_scores, pred_scores[index]])
                    processed_bboxes = np.concatenate(
                        [processed_bboxes, pred_bboxes[index]])
                    extend_labels = np.full(index.shape, cls, dtype=np.int64)
                    processed_labels = np.concatenate(
                        [processed_labels, extend_labels])
                    processed_masks.extend([pred_masks[k] for k in index])
                elif cls not in allowed_classes and self.filter_labels:
                    # remove unannotated preds
                    index = np.where(processed_labels != cls)[0]
                    processed_scores = processed_scores[index]
                    processed_bboxes = processed_bboxes[index]
                    processed_labels = processed_labels[index]
                    processed_masks = [processed_masks[k] for k in index]
        return processed_bboxes, processed_scores, processed_labels, processed_masks
    
    def xyxy2xywh(self, bbox: np.ndarray) -> list:
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox: List = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def results2json(self, results: Sequence[dict],
                     outfile_prefix: str) -> dict:
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict: Possible keys are "bbox", "segm", "proposal", and
            values are corresponding filenames.
        """
        # import ipdb
        # ipdb.set_trace()
        bbox_json_results = []
        segm_json_results = [] if 'masks' in results[0] else None
        for idx, result in enumerate(results):
            image_id = result.get('img_id', idx)
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            # bbox results
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(scores[i])
                data['category_id'] = self.cat_ids[label]
                bbox_json_results.append(data)

            if segm_json_results is None:
                continue

            # segm results
            masks = result['masks']
            mask_scores = result.get('mask_scores', scores)
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(mask_scores[i])
                data['category_id'] = self.cat_ids[label]
                if isinstance(masks[i]['counts'], bytes):
                    masks[i]['counts'] = masks[i]['counts'].decode()
                data['segmentation'] = masks[i]
                segm_json_results.append(data)

        result_files = dict()
        result_files['bbox'] = f'{outfile_prefix}.bbox.json'
        result_files['proposal'] = f'{outfile_prefix}.bbox.json'
        dump(bbox_json_results, result_files['bbox'])

        if segm_json_results is not None:
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            dump(segm_json_results, result_files['segm'])

        return result_files

    def gt_to_coco_json(self, gt_dicts: Sequence[dict],
                        outfile_prefix: str) -> str:
        """Convert ground truth to coco format json file.

        Args:
            gt_dicts (Sequence[dict]): Ground truth of the dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json file will be named
                "somepath/xxx.gt.json".
        Returns:
            str: The filename of the json file.
        """
        categories = [
            dict(id=id, name=name)
            for id, name in enumerate(self.dataset_meta['classes'])
        ]
        image_infos = []
        annotations = []

        for idx, gt_dict in enumerate(gt_dicts):
            img_id = gt_dict.get('img_id', idx)
            image_info = dict(
                id=img_id,
                width=gt_dict['width'],
                height=gt_dict['height'],
                file_name='')
            image_infos.append(image_info)
            for ann in gt_dict['anns']:
                label = ann['bbox_label']
                bbox = ann['bbox']
                coco_bbox = [
                    bbox[0],
                    bbox[1],
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                ]

                annotation = dict(
                    id=len(annotations) +
                    1,  # coco api requires id starts with 1
                    image_id=img_id,
                    bbox=coco_bbox,
                    iscrowd=ann.get('ignore_flag', 0),
                    category_id=int(label),
                    area=coco_bbox[2] * coco_bbox[3])
                if ann.get('mask', None):
                    mask = ann['mask']
                    # area = mask_util.area(mask)
                    if isinstance(mask, dict) and isinstance(
                            mask['counts'], bytes):
                        mask['counts'] = mask['counts'].decode()
                    annotation['segmentation'] = mask
                    # annotation['area'] = float(area)
                annotations.append(annotation)

        info = dict(
            date_created=str(datetime.datetime.now()),
            description='Coco json file converted by mmdet CocoMetric.')
        coco_json = dict(
            info=info,
            images=image_infos,
            categories=categories,
            licenses=None,
        )
        if len(annotations) > 0:
            coco_json['annotations'] = annotations
        converted_json_path = f'{outfile_prefix}.gt.json'
        dump(coco_json, converted_json_path)
        return converted_json_path

    # TODO: data_batch is no longer needed, consider adjusting the
    #  parameter position
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        # import ipdb
        # ipdb.set_trace()
        for data_sample in data_samples:
            data_sample_copy = copy.deepcopy(data_sample)
            # add super-category instances
            # TODO: Need to refactor to support LoadAnnotations
            instances = data_sample_copy['instances']
            if self.get_supercategory:
                supercat_instances = self._get_supercategory_ann(instances)
                instances.extend(supercat_instances)
            gt = dict()
            gt['width'] = data_sample_copy['ori_shape'][1]
            gt['height'] = data_sample_copy['ori_shape'][0]
            gt['img_id'] = data_sample_copy['img_id']
            # TODO: Need to refactor to support LoadAnnotations
            if self._coco_api is None:
                # TODO: Need to refactor to support LoadAnnotations
                assert 'instances' in data_sample, \
                    'ground truth is required for evaluation when ' \
                    '`ann_file` is not provided'
                gt['anns'] = instances

            image_level_labels = data_sample_copy.get('image_level_labels', None)
            pred = data_sample_copy['pred_instances']
            pred_bboxes = pred['bboxes'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()
            pred_masks = encode_mask_results(
                pred['masks'].detach().cpu().numpy()) if isinstance(
                    pred['masks'], torch.Tensor) else pred['masks']

            pred_bboxes, pred_scores, pred_labels, pred_masks = self._process_predictions(
                pred_bboxes, pred_scores, pred_labels, pred_masks, instances,
                image_level_labels)

            result = dict()
            result['img_id'] = data_sample_copy['img_id']
            result['bboxes'] = pred_bboxes
            result['scores'] = pred_scores
            result['labels'] = pred_labels
            result['masks'] = pred_masks
            self.results.append((gt,result))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # split gt and prediction list
        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        if self._coco_api is None:
            # use converted gt json file to initialize coco api
            logger.info('Converting ground truth to coco format...')
            coco_json_path = self.gt_to_coco_json(
                gt_dicts=gts, outfile_prefix=outfile_prefix)
            self._coco_api = COCO(coco_json_path)

        # handle lazy init
        if self.cat_ids is None:
            self.cat_ids = self._coco_api.get_cat_ids(
                cat_names=self.dataset_meta['classes'])
        if self.img_ids is None:
            self.img_ids = self._coco_api.get_img_ids()

        # convert predictions to coco format and dump to json file
        result_files = self.results2json(preds, outfile_prefix)

        eval_results = OrderedDict()
        if self.format_only:
            logger.info('results are saved in '
                        f'{osp.dirname(outfile_prefix)}')
            return eval_results

        for metric in self.metrics:
            logger.info(f'Evaluating {metric}...')


            # evaluate proposal, bbox and segm
            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                    '''
                    {'image_id': 289393, 'bbox': [230.8678741455078, 184.56314086914062, 407.8108367919922, 252.25137329101562], 'score': 0.05109857767820358, 'category_id': 1, 'segmentation': {'size': [480, 640], 'counts': '
                    '''
                coco_dt = self._coco_api.loadRes(predictions)

            except IndexError:
                logger.error(
                    'The testing results of the whole dataset is empty.')
                break

            coco_eval = COCOeval(self._coco_api, coco_dt, iou_type)

            coco_eval.params.catIds = self.cat_ids
            coco_eval.params.imgIds = self.img_ids
            # coco_eval.params.maxDets = list(self.proposal_nums)
            coco_eval.params.iouThrs = self.iou_thrs

            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            metric_items = self.metric_items
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item "{metric_item}" is not supported')

            if metric == 'proposal':
                coco_eval.params.useCats = 0
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{coco_eval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                if self.classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = coco_eval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, cat_id in enumerate(self.cat_ids):
                        t = []
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self._coco_api.loadCats(cat_id)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        t.append(f'{nm["name"]}')
                        t.append(f'{round(ap, 3)}')
                        eval_results[f'{nm["name"]}_precision'] = round(ap, 3)

                        # indexes of IoU  @50 and @75
                        for iou in [0, 5]:
                            precision = precisions[iou, :, idx, 0, -1]
                            precision = precision[precision > -1]
                            if precision.size:
                                ap = np.mean(precision)
                            else:
                                ap = float('nan')
                            t.append(f'{round(ap, 3)}')

                        # indexes of area of small, median and large
                        for area in [1, 2, 3]:
                            precision = precisions[:, :, idx, area, -1]
                            precision = precision[precision > -1]
                            if precision.size:
                                ap = np.mean(precision)
                            else:
                                ap = float('nan')
                            t.append(f'{round(ap, 3)}')
                        results_per_category.append(tuple(t))

                    num_columns = len(results_per_category[0])
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = [
                        'category', 'mAP', 'mAP_50', 'mAP_75', 'mAP_s',
                        'mAP_m', 'mAP_l'
                    ]
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    logger.info('\n' + table.table)

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = coco_eval.stats[coco_metric_names[metric_item]]
                    eval_results[key] = float(f'{round(val, 3)}')

                ap = coco_eval.stats[:6]
                logger.info(f'{metric}_mAP_copypaste: {ap[0]:.3f} '
                            f'{ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                            f'{ap[4]:.3f} {ap[5]:.3f}')

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results