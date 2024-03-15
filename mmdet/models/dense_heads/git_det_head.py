# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList
from mmdet.utils import (ConfigType, InstanceList, OptInstanceList,
                         OptMultiConfig, reduce_mean, InstanceList, OptInstanceList)
from ..utils import multi_apply
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh

@MODELS.register_module()
class GiTDetHead(BaseModule):
    r"""Detection head for GiT. It's a non-parametric head for 
        GiT decoding and post-processing in detection task.
    """
    def __init__(self,
            train_cfg: ConfigType = None,
            test_cfg: ConfigType = None,
            init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided when train_cfg is set.'
            assigner = train_cfg['assigner']
            self.assigner = TASK_UTILS.build(assigner)
            if train_cfg.get('sampler', None) is not None:
                raise RuntimeError('GiT do not build sampler.')
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        pass

    def init_weights(self) -> None:
        pass

    def reset_hyparameter(self, cfgs):
        for k in list(cfgs.keys()):
            setattr(self, k, cfgs[k])
        self.loss_reg = nn.CrossEntropyLoss(reduction='sum', ignore_index=self.num_vocal)
    
    def get_targets_based_on_reference(self, 
                reference_preds_list: List[Tensor],
                batch_gt_instances: InstanceList,
                batch_img_metas: List[dict]) -> tuple:
        """Compute regression and classification targets for a batch image.

        Assign targets based on distance between boxes and grids.

        Args:
            reference_preds_list (list[Tensor]): Grid positions for each image, 
                with normalized coordinate (cx, cy) and shape [num_queries, 2].
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            tuple: a tuple containing the following targets.

            - input_tokens_tensor (Tensor): Input tokens of each image for training.
              has shape (bs, num_queries, 4).
            - targets_tokens_tensor (Tensor): GT tokens of each image (bs, num_queries, 5).
            - tokens_weights_tensor (Tensor): GT tokens weights of each image, 
              has shape (bs, num_queries, 5).
            - num_total_pos (int): Number of positive samples in all images.
            - num_total_neg (int): Number of negative samples in all images.
        """
        (input_tokens_list, targets_tokens_list, tokens_weights_list, pos_inds_list, 
         neg_inds_list) = multi_apply(self._get_targets_single_based_on_reference,
                                      reference_preds_list,
                                      batch_gt_instances, batch_img_metas)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))

        # only support parallel training, means query_num of each image is equal
        return (torch.stack(input_tokens_list), torch.stack(targets_tokens_list), 
                torch.stack(tokens_weights_list), num_total_pos, num_total_neg)
    
    def _get_targets_single_based_on_reference(self, reference_pred: Tensor,
                gt_instances: InstanceData,
                img_meta: dict) -> tuple:
        """Compute regression and classification targets for one image.

        Assign targets based on distance between boxes and grids.

        Args:
            reference_pred (Tensor): Grid positions for one image, 
                with normalized coordinate (cx, cy) and shape [num_queries, 2]
                or normalized coordinate (cx, cy, w, h) and shape [num_queries, 4].
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - input_tokens (Tensor): Input tokens of each image for training.
            - targets_tokens (Tensor): GT tokens of each image.
            - tokens_weights (Tensor]): GT tokens weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        img_h, img_w = img_meta['img_shape']
        if reference_pred.shape[-1] == 2:
            # cx, cy
            factor = reference_pred.new_tensor([img_w, img_h]).unsqueeze(0)
            # convert reference_pred from normalized to unnormalized
            reference_pred = reference_pred * factor
        elif reference_pred.shape[-1] == 4:
            # cx, cy, w, h
            factor = reference_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
            # convert reference_pred from normalized to unnormalized
            reference_pred = bbox_cxcywh_to_xyxy(reference_pred) * factor
        else:
            raise NotImplementedError
        num_bboxes = reference_pred.size(0)
        reference_pred_instances = InstanceData(points=reference_pred)
        # assigner and sampler
        assign_result = self.assigner.assign(
            pred_instances=reference_pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta)
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        # get postive gt
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds.long(), :]
        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)
        # bbox targets
        bboxes = gt_bboxes.new_full((reference_pred.shape[0], 4), self.num_vocal - 1, dtype=torch.float)
        bbox_weights = gt_bboxes.new_zeros((reference_pred.shape[0], 4))
        bbox_weights[pos_inds] = 1.0

        # GiT regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        pos_gt_bboxes_normalized = pos_gt_bboxes / reference_pred.new_tensor([
                                    img_w, img_h, img_w, img_h]).unsqueeze(0)
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)

        # center prediction is converted to center residual
        pos_reference_pred_normalized = \
                reference_pred[pos_inds] / reference_pred.new_tensor([
                                            img_w, img_h]).unsqueeze(0)
        pos_center_residual = pos_gt_bboxes_targets[:, :2] - pos_reference_pred_normalized
        # assume pos_center_residual range [-0.5, 0.5]
        pos_gt_bboxes_targets[:, :2] = pos_center_residual + 0.5
                
        bboxes[pos_inds] = pos_gt_bboxes_targets # [0., 1.]

        # convert to gt to tokens
        # coord and scale token [0, self.num_bins]
        bboxes_tokens = (bboxes * self.num_bins).round().long().clamp(min=0, max=self.num_bins)

        # ignore 
        bboxes_tokens[neg_inds] = self.num_vocal - 1
        # classification token
        class_targets_tokens = labels + self.num_bins + 1
        targets_tokens = torch.cat([class_targets_tokens[:, None], bboxes_tokens], dim=1)
        tokens_weights = torch.cat([label_weights[:, None], bbox_weights], dim=1)

        # input tokens for parallel training
        input_tokens = targets_tokens[:, :-1]

        return (input_tokens, targets_tokens, tokens_weights, pos_inds, neg_inds)
    
    def loss(self, all_layer_pred_seq_logits: Tensor,
                   all_layer_target_tokens: List[Tensor],
                   all_layer_token_weights: List[Tensor],
                   num_total_pos: List[int], 
                   num_total_neg: List[int],
                   batch_data_samples: SampleList) -> Dict[str, Tensor]:
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        loss_inputs = (all_layer_pred_seq_logits,
                       all_layer_target_tokens,
                       all_layer_token_weights,
                       num_total_pos,
                       num_total_neg,
                       batch_gt_instances, batch_img_metas)
        losses = self.loss_by_feat(*loss_inputs)
        return losses
    
    def loss_by_feat(self, all_layer_pred_seq_logits: Tensor,
                           all_layer_target_tokens: List[Tensor],
                           all_layer_token_weights: List[Tensor],
                           num_total_pos: List[int], 
                           num_total_neg: List[int],
                           batch_gt_instances: InstanceList,
                           batch_img_metas: List[dict],
                           batch_gt_instances_ignore: OptInstanceList = None) -> Dict[str, Tensor]:
        """"Loss function.

        Only outputs from the last feature level are used for computing
        losses by default.

        Args:
            all_layer_pred_seq_logits (Tensor): Outputs from the
                autoregressive head, has shape (num_decoder_layers, bs,
                num_queries, max_token_len, vocab_size).
            all_layer_target_tokens (Tensor): GT targets for
                autoregressive head, has shape (num_decoder_layers, bs,
                num_queries, max_token_len).
            all_layer_token_weights (Tensor): GT weights of 
                each token, has shape (num_decoder_layers, bs, num_queries, 
                max_token_len).
            num_total_pos (List[int]): Number of positive samples in all images.
            num_total_neg (List[int]): Number of negative samples in all images.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert batch_gt_instances_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            'for batch_gt_instances_ignore setting to None.'
        losses_cls, losses_reg= multi_apply(
            self.loss_by_feat_single,
            all_layer_pred_seq_logits,
            all_layer_target_tokens,
            all_layer_token_weights,
            num_total_pos, 
            num_total_neg,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas)

        loss_dict = dict()
        # loss from the last decoder layer
    
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_reg'] = losses_reg[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_reg_i in zip(losses_cls[:-1], losses_reg[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i[0]
            loss_dict[f'd{num_dec_layer}.loss_reg'] = loss_reg_i[0]
            num_dec_layer += 1
        return loss_dict

    def loss_by_feat_single(self, pred_seq_logits: Tensor, 
                                  targets_tokens_tensor: Tensor,
                                  tokens_weights_tensor: Tensor,
                                  num_total_pos: int, 
                                  num_total_neg: int,
                                  batch_gt_instances: InstanceList,
                                  batch_img_metas: List[dict]) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            pred_seq_logits (Tensor): Outputs from the autoregressive head, 
                has shape (bs, num_queries, max_token_len, vocab_size).
            targets_tokens_tensor (Tensor): GT targets for autoregressive head, 
                has shape (bs, num_queries, max_token_len).
            tokens_weights_tensor (Tensor): GT weights of each token, has shape 
                (bs, num_queries, max_token_len).
            num_total_pos (int): Number of positive samples in all images.
            num_total_neg (int): Number of negative samples in all images.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls` and `loss_reg`
        """
        num_imgs, num_queries = pred_seq_logits.shape[:2]

        # split classification and regression logits
        pred_seq_cls_logits = pred_seq_logits[:, :, 0, self.num_bins + 1:].reshape(-1, self.num_classes + 1)
        pred_seq_reg_logits = pred_seq_logits[:, :, 1:, :self.num_bins + 1].reshape(-1, self.num_bins + 1)
        # construct weighted avg_factor 
        # every query need classfication, only postive query need regression
        avg_factor = num_imgs * num_queries + \
            num_total_pos * (self.dec_length - 1)
        avg_factor = reduce_mean(
            pred_seq_logits.new_tensor([avg_factor]))
        avg_factor = max(avg_factor, 1)

        # ignore negative queries regression
        tokens_weights_tensor = tokens_weights_tensor.view(-1)
        targets_tokens_tensor = targets_tokens_tensor.view(-1)
        ignore_token_ids = torch.nonzero((tokens_weights_tensor==0.)).squeeze(-1)
        targets_tokens_tensor[ignore_token_ids] = self.num_vocal

        # split classfication and regression targets
        cls_targets_tokens_tensor = targets_tokens_tensor.view(-1, 5)[:, 0] - self.num_bins - 1
        reg_targets_tokens_tensor = targets_tokens_tensor.view(-1, 5)[:, 1:].reshape(-1)

        loss_cls = self.loss_reg(pred_seq_cls_logits, cls_targets_tokens_tensor) / avg_factor
        loss_reg = self.loss_reg(pred_seq_reg_logits, reg_targets_tokens_tensor) / avg_factor

        return (loss_cls, loss_reg)
    
    def decoder_inference(self, layers_module, patch_embed: Tensor, patch_mask: Tensor, text_embed: Tensor, text_mask: Tensor, 
            grid_pos_embed: Tensor, grid_mask: Tensor, references: Tensor, bert_embed_func: Callable, task_embedding: Tensor, 
            vocabulary_embed: Tensor, grid_interpolate: bool=True, global_only_image: bool=True) -> Dict:
        """AutoRegressive decoding target tokens.
        
        Args:
            layers_module (torch module): transformer module with parameter.
            patch_embed (Tensor): image patch embedding has (bs, patch_H, patch_W, C).
            patch_mask (Tensor): image patch mask has (bs, patch_H, patch_W).
            text_embed (Tensor): text input embedding. Default is None.
            text_mask (Tensor): text input mask. Default is None.
            grid_pos_embed (Tensor): grid_pos_embed has (bs, sampled_query_num, C).
                task identifier + position embedding.
            grid_mask (Tensor): grid mask has (bs, grid_H, grid_W)
            references (Tensor): normalized grid position (bs, num_queries, 2).
            bert_embed_func (Callable): bert embedding function.
            task_embedding (Tensor): task identifier embedding for each task with shape (C)
            vocabulary_embed (Tensor): dynamic vocabulary for this task with (vocabulary_num, C)
            grid_interpolate (bool): if use grid interpolation for local information. Default is True.
            global_only_image (bool): if global layer only process image. Default is True.

        Returns:
            dict: The dictionary of decoding outputs.
        """
        pre_kv_list = []
        patch_resolution = patch_embed.shape[1:3]
        grid_resolution_perwin = [grid_mask.shape[i+1] // (patch_resolution[i] \
                    // layers_module[0].window_size) for i in range(2)]
        batch_size, query_num = references.shape[:2]
        references = references[:, :, :2]
        image_patch = patch_embed 
        grid_token = grid_pos_embed.clone() 
        grid_mask = grid_mask.flatten(1)

        # compute observation interaction (e.g., image, text, and local feature token)
        grid_interpolate_feats = [] if grid_interpolate else None
        for layer_id, layer in enumerate(layers_module):
            if grid_interpolate:
                # compute tokens of local image feature
                input_img_patch = image_patch.permute(0, 3, 1, 2) 
                grid_position = references[:, :, :2].unsqueeze(2) * 2 - 1 
                grid_local_feat = F.grid_sample(input_img_patch, grid_position, align_corners=False)
                grid_interpolate_feats.append(grid_local_feat.squeeze(-1).permute(0, 2, 1))
            # window-based local forward passing or global forward passing
            window_patch_num = layer.window_size ** 2 if layer.window_size > 0 else np.prod(patch_resolution)
            scope_len = window_patch_num
            # if the input includes text observation
            if text_embed is not None:
                observe_num = window_patch_num + text_embed.shape[1]
                scope_len += text_embed.shape[1]
                attn_mask = torch.zeros(scope_len, scope_len, device=grid_token.device)
                # mask text for autoregressive manner
                text_len = observe_num - window_patch_num
                attn_mask[window_patch_num:observe_num, window_patch_num:observe_num] = \
                    torch.triu(torch.ones(text_len, text_len, device=grid_token.device), diagonal=1)
            else:
                observe_num = window_patch_num
                attn_mask = torch.zeros(scope_len, scope_len, device=grid_token.device)

            image_patch, text_embed, inter_kv = layer.img_forward(image_patch, text_embed,  
                            attn_mask[:observe_num, :observe_num].bool(), patch_mask, text_mask,return_intermediate=True)
            pre_kv_list.append(inter_kv)
        
        outputs_coords = []

        for pos_id in range(0, self.dec_length + 1):
            # initial token embedding of the first layer for each predictive position
            if pos_id == 0:
                # local image information token
                input_embed = grid_token.view(batch_size * query_num, 1, -1)
            elif pos_id == 1:
                # task identifier token
                input_embed = bert_embed_func(inputs_embeds=task_embedding[None, None, :].repeat(batch_size * query_num, 1, 1))
            x = input_embed
            # decoder each layer
            for layer_id, layer in enumerate(layers_module):
                if layer.window_size <= 0 and global_only_image:
                    # in this case, only input interaction without input-output interaction
                    continue
                x = x.view(batch_size, query_num, 1, -1)
                # update local image token for the first token of each layer
                if grid_interpolate and pos_id == 0:
                    x += grid_interpolate_feats[layer_id].unsqueeze(2)
                # pos embedding is re-added to the first token before each layer computation begins
                if layer_id > 0 and pos_id == 0:
                    x += grid_pos_embed.view(batch_size, query_num, 1, -1)
                
                # generate attn masks
                window_patch_num = layer.window_size ** 2 if layer.window_size > 0 else np.prod(patch_resolution)
                unit_grid_num = np.prod(grid_resolution_perwin) if layer.window_size > 0 else query_num
                observe_num = window_patch_num + text_embed.shape[1] if text_embed is not None else window_patch_num 
                attn_mask = torch.zeros(unit_grid_num, observe_num, device=input_embed.device)
                iter_pad_masks = (1. - torch.eye(unit_grid_num).to(attn_mask.device)).repeat(1, pos_id+1)
                attn_mask = torch.cat([attn_mask, iter_pad_masks], dim=1)

                # target attention computation
                x, pre_kv_update = layer.token_forward(image_patch=image_patch, grid_token=x, grid_position=references, idx=pos_id,
                    attn_mask=attn_mask.bool(), patch_mask=patch_mask, grid_mask=grid_mask, text_mask=text_mask,  pre_kv=pre_kv_list[layer_id])

                pre_kv_list[layer_id] = pre_kv_update
                x = x.view(batch_size*query_num, 1, -1)
                
                # decode target based on argmax
                if pos_id == 0:
                    continue
                if layer_id == (len(layers_module)-1):
                    logits = (x @ vocabulary_embed.transpose(0, 1))[:, -1, :]
                    if pos_id == 1:  # class label prediction (soft classes logits)
                        start_offset = self.num_bins + 1
                        current_logits = logits[:, start_offset: start_offset + self.num_classes + 1]
                        outputs_classes_logits = current_logits.softmax(dim=-1)[:, :-1]
                        pred_token = torch.argmax(current_logits, dim=-1, keepdim=True)
                        outputs_classes = outputs_classes_logits
                    elif pos_id > 1 and pos_id <= self.dec_length:  # bbox prediction (hard label)
                        start_offset = 0
                        current_logits = logits[:, :self.num_bins+1]
                        pred_token = torch.argmax(current_logits, dim=-1, keepdim=True)
                        outputs_coords.append(pred_token)
                    else:
                        raise RuntimeError('detection only contains 5 tokens.')

            if pos_id > 0:
                input_embed = bert_embed_func(inputs_embeds=vocabulary_embed[(pred_token + start_offset)], 
                                         past_key_values_length=pos_id)

        outputs_classes = outputs_classes.view(batch_size, query_num, -1)
        outputs_coords = torch.cat(outputs_coords, dim=-1).view(batch_size, query_num, -1) / self.num_bins
        # residual center regression
        outputs_coords[:, :, :2] = outputs_coords[:, :, :2] - 0.5 + references
        output_dict = {'outputs_classes': outputs_classes, 'outputs_coords': outputs_coords}
            
        return output_dict
    
    def predict(self, outputs_classes: Tensor, outputs_coords: Tensor,
            batch_data_samples: SampleList, rescale: bool = True) -> InstanceList:
        """Perform inference of detection head.

        Args:
            outputs_classes (Tensor): Classification scores of the last layer, 
                has shape (bs, num_queries, cls_out_channels).
            outputs_coords (Tensor): Regression outputs of the last layers. 
                Each is a 3D-tensor with normalized coordinate format
                (cx, cy, w, h) and shape (bs, num_queries, 4).
            batch_data_samples (list[:obj:`DetDataSample`]): The Data
                 Samples. It usually includes information such as
                 `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): If `True`, return boxes in original image space. 

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        batch_img_metas = [data_samples.metainfo for data_samples in batch_data_samples]
        result_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score = outputs_classes[img_id]
            bbox_pred = outputs_coords[img_id]
            img_meta = batch_img_metas[img_id]
            results = self._predict_single(cls_score, bbox_pred, img_meta, rescale)
            result_list.append(results)
        return result_list

    def _predict_single(self, cls_score: Tensor, bbox_pred: Tensor,
                    img_meta: dict, rescale: bool = True) -> InstanceData:
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Argmax outputs from the last layer for each image, 
                with coordinate format (cx, cy, w, h) and shape [num_queries, 4].
            img_meta (dict): Image meta info.
            rescale (bool): If True, return boxes in original image space. 

        Returns:
            :obj:`InstanceData`: Detection results of each image after the post process.
            Each item usually contains following keys.
                - scores (Tensor): Classification scores, has a shape (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4), the last dimension 4 
                  arrange as (x1, y1, x2, y2).
        """
        max_per_img = self.test_cfg.get('max_per_img', len(cls_score))
        # NOTE: assume that all the images are in the same scale 
        img_shape = img_meta['img_shape'] # or img_meta['batch_input_shape']

        scores, indexes = cls_score.reshape(-1).topk(max_per_img)
        det_labels = indexes % self.num_classes
        bbox_index = indexes // self.num_classes
        bbox_pred = bbox_pred[bbox_index]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            assert img_meta.get('scale_factor') is not None
            det_bboxes /= det_bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))
        results = InstanceData()
        
        results.bboxes = det_bboxes
        results.scores = scores
        results.labels = det_labels
        return results

