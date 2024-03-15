# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.registry import MODELS

class LayerNorm2d(nn.LayerNorm):
    """LayerNorm on channels for 2d images.

    Args:
        num_channels (int): The number of channels of the input tensor.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-5.
        elementwise_affine (bool): a boolean value that when set to ``True``,
            this module has learnable per-element affine parameters initialized
            to ones (for weights) and zeros (for biases). Defaults to True.
    """

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]

    def forward(self, x, data_format='channel_first'):
        """Forward method.

        Args:
            x (torch.Tensor): The input tensor.
            data_format (str): The format of the input tensor. If
                ``"channel_first"``, the shape of the input tensor should be
                (B, C, H, W). If ``"channel_last"``, the shape of the input
                tensor should be (B, H, W, C). Defaults to "channel_first".
        """
        assert x.dim() == 4, 'LayerNorm2d only supports inputs with shape ' \
            f'(N, C, H, W), but got tensor with shape {x.shape}'
        if data_format == 'channel_last':
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias,
                             self.eps)
        elif data_format == 'channel_first':
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias,
                             self.eps)
            # If the output is discontiguous, it may cause some unexpected
            # problem in the downstream tasks
            x = x.permute(0, 3, 1, 2).contiguous()
        return x


def build_norm_layer(cfg: dict, num_features: int) -> nn.Module:
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.

        num_features (int): Number of input channels.

    Returns:
        nn.Module: The created norm layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    norm_layer = MODELS.get(layer_type)
    if norm_layer is None:
        raise KeyError(f'Cannot find {layer_type} in registry under scope '
                       f'name {MODELS.scope}')

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)

    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
    else:
        layer = norm_layer(num_channels=num_features, **cfg_)

    if layer_type == 'SyncBN' and hasattr(layer, '_specify_ddp_gpu_num'):
        layer._specify_ddp_gpu_num(1)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return layer
