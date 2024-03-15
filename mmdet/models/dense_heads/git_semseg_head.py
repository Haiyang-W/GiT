# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mmcv
from mmengine.model import BaseModule
from mmengine.structures import InstanceData, PixelData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList, SegDataSample
from mmdet.utils import (ConfigType, InstanceList, OptInstanceList, PixelList, OptPixelList,
                         OptMultiConfig, reduce_mean, InstanceList, OptInstanceList,
                         SegOptSampleList, SegSampleList)
from ..utils import multi_apply, resize
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy

@MODELS.register_module()
class GiTSemSegHead(BaseModule):
    r"""Semantic Segmentation head for GiT. It's a non-parametric head for
        GiT decoding and post-processing in semantic segmentation task.
    """
    def __init__(self,
            init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self._init_layers()

    def _init_layers(self) -> None:
        pass

    def init_weights(self) -> None:
        pass
    
    def reset_hyparameter(self, cfgs):
        for k in list(cfgs.keys()):
            setattr(self, k, cfgs[k])
        self.loss_cls = nn.CrossEntropyLoss(reduction='sum', ignore_index=self.num_vocal)
    
    def get_targets_based_on_reference(self, 
                reference_preds_list: List[Tensor],
                batch_gt_instances: PixelList,
                batch_img_metas: List[dict]) -> tuple:
        """Compute semantic targets for a batch image.

        Args:
            reference_preds_list (list[Tensor]): Grid positions for each image, 
                with normalized coordinate (cx, cy) and shape [num_queries, 2].
            batch_gt_instances (list[:obj:`PixelData`]): Batch of
                gt_sem_seg. It usually includes semantic mask annotation.
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
                            gt_instances: PixelData, img_meta: dict) -> tuple:
        """Compute segmantic targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            reference_pred (Tensor): Grid positions for one image, 
                with normalized coordinate (cx, cy) and shape [num_queries, 2]
                or normalized coordinate (cx, cy, w, h) and shape [num_queries, 4].
            gt_instances (:obj:`PixelData`): Semantic ground truth of pixel annotations. 
            img_meta (dict): Meta information for one image.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - input_tokens (Tensor): Input tokens of each image for training.
            - targets_tokens (Tensor): GT tokens of each image.
            - tokens_weights (Tensor]): GT tokens weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        img_h, img_w = img_meta['pad_shape']
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
        
        # compute gt
        grid_h, grid_w = img_meta['grid_resolution']
        decode_resolution = (int(img_h / self.dec_pixel_resolution[0]),
                             int(img_w / self.dec_pixel_resolution[1]))
        assert decode_resolution[0] % grid_h == 0 and decode_resolution[1] % grid_w == 0, \
            'grids mush cover image uniformly, which means decode_hw must be divisible by grid_hw.'
        
        # local range for each grid
        win_h = int(decode_resolution[0] // grid_h)
        win_w = int(decode_resolution[1] // grid_w)
        assert win_h * win_w == self.dec_length, 'decode length is mismatched.'

        seg_label = mmcv.imrescale(gt_instances.data[0].cpu().numpy(), 
                decode_resolution, interpolation='nearest', backend='cv2')
        seg_label = gt_instances.data[0].new_tensor(seg_label) # decoder_h, decoder_w

        pixel_grid_y, pixel_grid_x = torch.meshgrid(
            torch.linspace(
                0, seg_label.shape[0] - 1, seg_label.shape[0], dtype=torch.int32, device=seg_label.device),
            torch.linspace(
                0, seg_label.shape[1] - 1, seg_label.shape[1], dtype=torch.int32, device=seg_label.device))
        pixel_grid = torch.cat([pixel_grid_x.unsqueeze(-1), pixel_grid_y.unsqueeze(-1)], -1)

        win_coord_W = pixel_grid[:, :, 0] // win_w
        win_coord_H = pixel_grid[:, :, 1] // win_h
        win_inds_eachpixel = win_coord_H * grid_w + win_coord_W
        win_inds_eachpixel = win_inds_eachpixel.int().view(-1)  # decode_h * decode_w

        inner_inds_w = pixel_grid[:, :, 0] % win_w
        inner_inds_h = pixel_grid[:, :, 1] % win_h
        inner_inds_eachpixel = inner_inds_h * win_w + inner_inds_w
        inner_inds_eachpixel = inner_inds_eachpixel.int().view(-1)

        global_inds = win_inds_eachpixel * win_h * win_w + inner_inds_eachpixel
        targets = seg_label.view(-1).clone()
        targets = targets.scatter_(0, global_inds.long(), seg_label.view(-1))
        # total_grid_num, decoder_len
        targets = targets.view(reference_pred.shape[0], win_h*win_w)

        # remove ignored region
        targets[targets == self.ignore_index] = self.num_vocal
        targets_tokens = targets
        tokens_weights = torch.ones_like(targets_tokens).float()

        # input tokens for parallel training
        input_tokens = targets_tokens[:, :-1]
        neg_inds = torch.nonzero(targets_tokens.view(-1) == (self.num_vocal)).squeeze(-1)
        pos_inds = torch.nonzero(targets_tokens.view(-1) != (self.num_vocal)).squeeze(-1)

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
            batch_gt_instances.append(data_sample.gt_sem_seg)

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
        losses_cls = multi_apply(
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
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, in zip(losses_cls[:-1],):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i[0]
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
            Tuple[Tensor]: A tuple including `loss_cls`
        """
        # classification loss
        pred_seq_cls_logits = pred_seq_logits[:, :, :, :self.num_vocal].reshape(-1, self.num_vocal)
        # construct weighted avg_factor 
        reg_avg_factor = num_total_pos
        reg_avg_factor = max(reg_avg_factor, 1)

        loss_cls = self.loss_cls(pred_seq_cls_logits, targets_tokens_tensor.view(-1)) / reg_avg_factor

        return (loss_cls, )
    
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
        grid_resolution = grid_mask.shape[1:]
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
            if text_embed is not None:
                observe_num = window_patch_num + text_embed.shape[1]
                scope_len += text_embed.shape[1]
                attn_mask = torch.zeros(scope_len, scope_len, device=grid_token.device)
                # mask text before
                text_len = observe_num - window_patch_num
                attn_mask[window_patch_num:observe_num,window_patch_num:observe_num] = \
                    torch.triu(torch.ones(text_len, text_len, device=grid_token.device), diagonal=1)
            else:
                observe_num = window_patch_num
                attn_mask = torch.zeros(scope_len, scope_len, device=grid_token.device)

            image_patch, text_embed, inter_kv = layer.img_forward(image_patch, text_embed,
                            attn_mask[:observe_num,:observe_num].bool(), patch_mask, text_mask, return_intermediate=True)
            pre_kv_list.append(inter_kv)
        
        outputs_classes = []

        for pos_id in range(0, self.dec_length + 1):
            # initial token embedding of the first layer for each predictive position
            if pos_id == 0:
                # local image information token
                input_embed = grid_token.view(batch_size * query_num, 1, -1)
            elif pos_id == 1:
                # task identifier token
                input_embed = bert_embed_func(inputs_embeds=task_embedding[None, None, :].repeat(batch_size * query_num, 1, 1), past_key_values_length=0)
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
                    attn_mask=attn_mask.bool(), pre_kv=pre_kv_list[layer_id], patch_mask=patch_mask, text_mask=text_mask, grid_mask=grid_mask)
                
                pre_kv_list[layer_id] = pre_kv_update
                x = x.view(batch_size*query_num, 1, -1)

                # decode target based on argmax
                if pos_id == 0:
                    continue
                if layer_id == (len(layers_module)-1):
                    logits = (x @ vocabulary_embed.transpose(0, 1))[:, -1, :self.num_vocal]
                    current_logits = logits
                    outputs_classes_logits = current_logits.softmax(dim=-1)[:, :-1]
                    outputs_classes_logits = outputs_classes_logits.view(batch_size, query_num, -1)
                    pred_token = torch.argmax(current_logits, dim=-1, keepdim=True)
                    outputs_classes.append(outputs_classes_logits)
            if pos_id > 0:
                input_embed = bert_embed_func(inputs_embeds=vocabulary_embed[pred_token], past_key_values_length=pos_id)

        outputs_classes = torch.stack(outputs_classes, dim=-2)
        output_dict = {'outputs_classes': outputs_classes, 'grid_resolution': grid_resolution}
        
        return output_dict
    
    def predict(self, outputs_classes: Tensor, grid_resolution: Tuple,
                      batch_data_samples: SampleList, rescale: bool = True) -> Tensor:
        """Transform grid decoding results to dense segmentation logits

        Args:
            - outputs_classes (Tensor): Outputs logits, has shape (bs,
                num_queries, decode_length, cls_out_channels).
            grid_resolution: (Tuple) grid resolution of whole image.
            batch_data_samples (list[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): If `True`, return boxes in original image space.

        Returns:
            seg_logits (Tensor): Dense semantic logits, has shape (bs, 
                cls_out_channels, h, w)
        """
        batch_img_metas = [data_samples.metainfo for data_samples in batch_data_samples]

        cls_scores = outputs_classes
        batch_size = len(batch_img_metas)
        assert batch_size == 1, 'only support batch size 1 in inference stage.'
        if 'pad_shape' in list(batch_img_metas[0].keys()):
            img_h, img_w = batch_img_metas[0]['pad_shape']
        else:
            img_h, img_w = batch_img_metas[0]['img_shape']

        # # reverse to semantic mask
        grid_h, grid_w = grid_resolution
        decode_resolution = (int(img_h / self.dec_pixel_resolution[0]),
                             int(img_w / self.dec_pixel_resolution[1]))
    
        assert decode_resolution[0] % grid_h == 0 and decode_resolution[1] % grid_w == 0, \
            'grids mush cover image uniformly, which means decode_hw must be divisible by grid_hw.'
        
        # local range for each grid
        win_h = int(decode_resolution[0] // grid_h)
        win_w = int(decode_resolution[1] // grid_w)
        assert win_h * win_w == self.dec_length, 'decode length is mismatched.'

        pixel_grid_y, pixel_grid_x = torch.meshgrid(
            torch.linspace(
                0, decode_resolution[0] - 1, decode_resolution[0], dtype=torch.int32, device=cls_scores.device),
            torch.linspace(
                0, decode_resolution[1] - 1, decode_resolution[1], dtype=torch.int32, device=cls_scores.device))
        pixel_grid = torch.cat([pixel_grid_x.unsqueeze(-1), pixel_grid_y.unsqueeze(-1)], -1)

        win_coord_W = pixel_grid[:, :, 0] // win_w
        win_coord_H = pixel_grid[:, :, 1] // win_h
        win_inds_eachpixel = win_coord_H * grid_w + win_coord_W
        win_inds_eachpixel = win_inds_eachpixel.int().view(-1)  # decode_h * decode_w

        inner_inds_w = pixel_grid[:, :, 0] % win_w
        inner_inds_h = pixel_grid[:, :, 1] % win_h
        inner_inds_eachpixel = inner_inds_h * win_w + inner_inds_w
        inner_inds_eachpixel = inner_inds_eachpixel.int().view(-1)
        
        # mapping relationship
        global_inds = win_inds_eachpixel * win_h * win_w + inner_inds_eachpixel
        global_inds = global_inds.long()

        cls_scores_flatten = cls_scores.view(cls_scores.shape[0], -1, cls_scores.shape[-1])
        cls_scores_refactor = cls_scores_flatten[:, global_inds, :]
        cls_scores_refactor = cls_scores_refactor.view(batch_size, *decode_resolution, 
                                                       cls_scores_refactor.shape[-1])
        seg_logits = cls_scores_refactor.permute(0, 3, 1, 2) # bchw
        return seg_logits
    
    def add_pred_to_datasample(self, data_samples: SegOptSampleList, 
                               seg_logits: Tensor) -> SegSampleList:
        """ Convert results list to `SegDataSample`.
        Args:
            seg_logits (Tensor): The segmentation results, seg_logits from
                model of each input image.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        batch_size, C, H, W = seg_logits.shape

        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                img_meta = data_samples[i].metainfo
                # remove padding area
                if 'img_padding_size' not in img_meta:
                    padding_size = img_meta.get('padding_size', [0] * 4)
                else:
                    padding_size = img_meta['img_padding_size']
                padding_left, padding_right, padding_top, padding_bottom =\
                    padding_size
                # i_seg_logits shape is 1, C, H, W after remove padding
                i_seg_logits = seg_logits[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]

                flip = img_meta.get('flip', None)
                if flip:
                    flip_direction = img_meta.get('flip_direction', None)
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        i_seg_logits = i_seg_logits.flip(dims=(3, ))
                    else:
                        i_seg_logits = i_seg_logits.flip(dims=(2, ))

                # resize as original shape
                i_seg_logits = resize(
                    i_seg_logits,
                    size=img_meta['ori_shape'],
                    mode='bilinear',
                    align_corners=False,
                    warning=False).squeeze(0)
            else:
                i_seg_logits = seg_logits[i]

            if C > 1:
                i_seg_pred = i_seg_logits.argmax(dim=0, keepdim=True)
            else:
                i_seg_logits = i_seg_logits.sigmoid()
                i_seg_pred = (i_seg_logits >
                              self.decode_head.threshold).to(i_seg_logits)
            data_samples[i].set_data({
                'seg_logits':
                PixelData(**{'data': i_seg_logits}),
                'pred_sem_seg':
                PixelData(**{'data': i_seg_pred})
            })

        return data_samples



