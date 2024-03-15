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
from mmdet.structures import SampleList, DataSample
from mmdet.utils import (ConfigType, InstanceList, OptInstanceList,
                         OptMultiConfig, reduce_mean, InstanceList, OptInstanceList)
from ..utils import multi_apply
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh

@MODELS.register_module()
class GiTCaptionHead(BaseModule):
    r"""Caption head for GiT. It's a non-parametric head for 
        GiT decoding and post-processing in image caption task.
    """
    def __init__(self,
            init_cfg: OptMultiConfig = None,
            temperature: float=0.7,
            alpha: float=0.75) -> None:
        super().__init__(init_cfg=init_cfg)
        self.temperature = temperature
        self.alpha = alpha

    def _init_layers(self) -> None:
        pass

    def init_weights(self) -> None:
        pass
    
    def reset_hyparameter(self, cfgs):
        for k in list(cfgs.keys()):
            setattr(self, k, cfgs[k])
        self.loss_cls = nn.CrossEntropyLoss(reduction='sum', ignore_index=self.ignore_index)

    def get_targets_based_on_reference(self, 
                reference_preds_list: List[Tensor],
                batch_gt_instances: InstanceList,
                batch_img_metas: List[dict],
                tokenizer: None) -> tuple:
        """Prepare next token targets for caption.

        Args:
            reference_preds_list (list[Tensor]): Grid positions for each image, 
                with normalized coordinate (cx, cy) and shape [num_queries, 2].
                Dummy for caption.
            batch_gt_instances (list[str]): Batch of
                gt_instance. It includes raw caption text.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            tokenizer: tokenizer class used in all tasks.
        Returns:
            tuple: a tuple containing the following targets.

            - input_tokens_tensor (Tensor): Input tokens of each image for training.
              has shape (bs, num_queries, dec_length).
            - target_tokens_tensor (Tensor): GT tokens of each image (bs, num_queries, dec_length).
            - tokens_weights_tensor (Tensor): GT tokens weights of each image, 
              has shape (bs, num_queries, dec_length).
            - num_total_pos (int): Number of positive samples in all images.
            - num_total_neg (int): Number of negative samples in all images.
        """
        batch_size = len(reference_preds_list)
        (input_tokens_list, target_tokens_list, tokens_weights_list, pos_inds_list, 
         neg_inds_list) = multi_apply(self._get_targets_single_based_on_reference,
                                reference_preds_list, batch_gt_instances, 
                                batch_img_metas, [tokenizer for _ in range(batch_size)])
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))

        # only support parallel training, means query_num of each image is equal
        return (torch.stack(input_tokens_list), torch.stack(target_tokens_list), 
                torch.stack(tokens_weights_list), num_total_pos, num_total_neg)
    
    def _get_targets_single_based_on_reference(self, reference_pred: Tensor,
                gt_instances: InstanceData,
                img_meta: dict,
                tokenizer: None) -> tuple:
        """Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            reference_pred (Tensor): Grid positions for each image, 
                with normalized coordinate (cx, cy) and shape [num_queries, 2] 
                or normalized coordinate (cx, cy, w, h) and shape [num_queries, 4].
                Dummy for caption.
            gt_instances (:obj:`str`): Ground truth of caption text
            img_meta (dict): Meta information for one image.
            tokenizer: tokenizer class used in all tasks.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - input_tokens (Tensor): Input tokens of each image for training.
            - target_tokens (Tensor): GT tokens of each image.
            - tokens_weights (Tensor]): GT tokens weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        # tokenizer raw text
        text = tokenizer(
            gt_instances,
            padding='max_length',
            truncation=True,
            max_length=self.dec_length,
            return_tensors='pt',
        ).to(reference_pred.device)
        text.input_ids[:, 0] = tokenizer.bos_token_id
        input_tokens = text.input_ids
        tokens_weights = text.attention_mask
        # not pred at end token
        tokens_weights[0,tokens_weights.sum()-1] = 0

        # prepare targets, ignore pad token and start token
        target_tokens = input_tokens.masked_fill(
            input_tokens == tokenizer.pad_token_id, self.ignore_index)
        target_tokens[...,:1] = self.ignore_index

        pos_inds = torch.nonzero(tokens_weights[0] > 0, as_tuple=False)
        neg_inds = torch.nonzero(tokens_weights[0] == 0, as_tuple=False)

        return (input_tokens, target_tokens, tokens_weights, pos_inds, neg_inds)
    
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
            batch_gt_instances.append(data_sample.gt_caption)

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
        for loss_cls_i in losses_cls[:-1]:
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i[0]
            num_dec_layer += 1
        return loss_dict

    def loss_by_feat_single(self, pred_seq_logits: Tensor, 
                            target_tokens_tensor: Tensor,
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
            target_tokens_tensor (Tensor): GT targets for autoregressive head, 
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
        # not pred at token
        pred_seq_cls_logits = pred_seq_logits[:, :, :-1,:].reshape(-1, self.num_vocal)
        # construct weighted avg_factor 
        cls_avg_factor = num_total_pos
        cls_avg_factor = reduce_mean(
            pred_seq_logits.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        # shift target to next token
        target_tokens_tensor = target_tokens_tensor[:,:,1:].contiguous()
    
        loss_cls = self.loss_cls(pred_seq_cls_logits, target_tokens_tensor.view(-1)) / cls_avg_factor

        return (loss_cls,)
    
    def decoder_inference(self, layers_module, patch_embed: Tensor, patch_mask: Tensor, text_embed: Tensor, text_mask: Tensor, 
            grid_pos_embed: Tensor, grid_mask: Tensor, references: Tensor, bert_embed_func: Callable, task_embedding: Tensor, 
            vocabulary_embed: Tensor, grid_interpolate: bool=True, global_only_image: bool=True, tokenizer: Callable=None) -> Dict:
        """AutoRegressive decoding target tokens.
        
        Args:
            layers_module (torch module): transformer module with parameter.
            patch_embed (Tensor): image patch embedding has (bs, patch_H, patch_W, C).
            patch_mask (Tensor): image patch mask has (bs, patch_H, patch_W).
            text_embed (Tensor): text input embedding. Default is None.
            text_mask (Tensor): text input mask. Default is None.
            grid_pos_embed (Tensor): grid_pos_embed has (bs, sampled_query_num, C).
                task identifier + position embedding.
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
        image_patch = patch_embed
        patch_resolution = patch_embed.shape[1:3]
        batch_size, query_num = image_patch.shape[0], 1
        T = self.temperature

        grid_token = grid_pos_embed.clone() 
        # compute observation interaction (e.g., image)

        assert not grid_interpolate, "caption doesn't need grid interpolation"
        for layer_id, layer in enumerate(layers_module):
            image_patch, _, inter_kv = layer.img_forward(image_patch, return_intermediate=True)
            pre_kv_list.append(inter_kv)
        
        pred_len = self.dec_length
        prev_beam_scores = torch.zeros((batch_size, self.beam_num),device=image_patch.device)
        beam_end_mask = torch.zeros((batch_size, self.beam_num),device=image_patch.device).bool()
        for pos_id in range(0, self.dec_length + 1):
            if pos_id == 0:
                # NOTE: for consistent with other task, acturally can be removed
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
                if layer_id > 0 and pos_id == 0:
                    x += grid_pos_embed.view(batch_size, query_num, 1, -1)
                
                # generate attn masks
                window_patch_num = layer.window_size ** 2 if layer.window_size > 0 else np.prod(patch_resolution)
                unit_grid_num = query_num
                attn_mask = torch.zeros(unit_grid_num, window_patch_num, device=input_embed.device)
                iter_pad_masks = (1. - torch.eye(unit_grid_num).to(attn_mask.device)).repeat(1, pos_id+1)
                attn_mask = torch.cat([attn_mask, iter_pad_masks], dim=1)

                x, pre_kv_update = layer.token_forward(image_patch=image_patch, grid_token=x, grid_position=None,
                    attn_mask=attn_mask.bool(), pre_kv=pre_kv_list[layer_id], disabled_window=True)

                pre_kv_list[layer_id] = pre_kv_update  
                
                # decode target based on argmax
                if pos_id == 0:
                    continue 
                if layer_id == (len(layers_module)-1):
                    x = x.view(batch_size*query_num, 1, -1)
                    logits = ((x @ vocabulary_embed.transpose(0, 1))[:, -1, :] / self.temperature).softmax(-1)
                    # beam search
                    if pos_id == 1:
                        prev_beam_scores, pred_token = torch.topk(logits, self.beam_num, dim=-1)
                        prev_beam_scores = torch.log(prev_beam_scores)
                        outputs_texts = pred_token.view(batch_size, self.beam_num,1)
                        query_num = self.beam_num
                        # repeat text token for beam num
                        new_pre_kv_list = []
                        for k in range(len(pre_kv_list)):
                            pre_kv = pre_kv_list[k]
                            image_kv, text_kv = pre_kv[:,:,:window_patch_num],pre_kv[:,:,window_patch_num:]
                            new_kv = torch.cat([image_kv,text_kv.unsqueeze(3).repeat(1,1,1, self.beam_num,1).flatten(2,3)],dim=2)
                            new_pre_kv_list.append(new_kv)
                        pre_kv_list = new_pre_kv_list    
                    else:
                        beam_logits = logits.view(batch_size,query_num,-1).log() + prev_beam_scores.unsqueeze(-1)
                        # set end beam always end token
                        end_idx = torch.nonzero(beam_end_mask)
                        end_token_idx = torch.cat([end_idx,torch.full((len(end_idx),1), tokenizer.sep_token_id, device=image_patch.device)],dim=-1)
                        for idx in end_token_idx:
                            a,b,c = idx
                            beam_logits[a,b,c] = 1e10
                        beam_logits = beam_logits.view(batch_size, -1)
                        
                        topk_logits, topk_ids = torch.topk(beam_logits, query_num,dim=-1)
                        pred_token = topk_ids % logits.shape[-1]
                        topk_beam = topk_ids // logits.shape[-1]

                        prev_beam_out = outputs_texts
                        # select beam in topk
                        prev_len = prev_beam_out.shape[-1]
                        topk_beam_out = torch.gather(prev_beam_out, 1, topk_beam.unsqueeze(-1).repeat(1, 1, prev_len))
                        outputs_texts = torch.cat([topk_beam_out, pred_token.unsqueeze(-1)], dim=-1)
                        # not add score after eos
                        beam_end_mask = (pred_token == tokenizer.sep_token_id)
                        prev_beam_scores[~beam_end_mask] = topk_logits[~beam_end_mask]

                        new_pre_kv_list = []
                        for k in range(len(pre_kv_list)):
                            pre_kv = pre_kv_list[k]
                            image_kv,text_kv = pre_kv[:, :, :window_patch_num],pre_kv[:, :, window_patch_num:]
                            num_heads = text_kv.shape[1] // batch_size
                            kv_dim = text_kv.shape[-1]
                            text_kv = text_kv.view(2,batch_size,num_heads, pos_id+1, self.beam_num,kv_dim).permute(1,4,0,2,3,5)
                            topk_text_kv = torch.gather(text_kv, 1, topk_beam.view(batch_size, self.beam_num, 1, 1, 1, 1 \
                                                    ).repeat(1, 1, 2, num_heads,pos_id+1, kv_dim))
                            topk_text_kv = topk_text_kv.permute(2, 0, 3, 4, 1, 5).flatten(1, 2).flatten(2, 3)
                            new_kv = torch.cat([image_kv,topk_text_kv],dim=2)
                            new_pre_kv_list.append(new_kv)
                        pre_kv_list = new_pre_kv_list
            if pos_id > 0:
                input_embed = bert_embed_func(input_ids=pred_token, past_key_values_length=pos_id)

        # normalize length
        out_len = (outputs_texts != tokenizer.sep_token_id).long().sum(dim=-1)
        prev_beam_scores = prev_beam_scores / torch.pow(out_len, self.alpha)
        _, max_beam_id = prev_beam_scores.max(dim=-1, keepdim=True)
        outputs_texts = torch.gather(outputs_texts, 1, max_beam_id.unsqueeze(-1).repeat(1, 1, pred_len)).squeeze(1)
        outputs_texts = tokenizer.batch_decode(outputs_texts, skip_special_tokens=True)
        output_dict = {'outputs_texts': outputs_texts}

        return output_dict
    
    def predict(self, outputs_texts: Tensor, batch_data_samples: SampleList, rescale: bool = True) -> List:
        return outputs_texts
