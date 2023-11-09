"""
Part of this code stems from rwightman's MaxVit implementation:
https://github.com/huggingface/pytorch-image-models/blob/1885bdc4318cc3be459981ea1a26cd862220864d/timm/models/maxxvit.py
that is:
- LayerScale
- PartitionAttentionCl
- window*
- grid*
- SelfAttentionCl
"""

from enum import Enum, auto
from functools import partial
from typing import Optional, Union, Tuple, List, Type

import math
import torch
from torch import nn

# from .layers import DropPath, LayerNorm
# from .layers import get_act_layer, get_norm_layer
# from .layers import to_2tuple, _assert


class PartitionType(Enum):
    WINDOW = auto()
    GRID = auto()


def nChw_2_nhwC(x: torch.Tensor):
    """N C H W -> N H W C
    """
    assert x.ndim == 4
    return x.permute(0, 2, 3, 1)


def nhwC_2_nChw(x: torch.Tensor):
    """N H W C -> N C H W
    """
    assert x.ndim == 4
    return x.permute(0, 3, 1, 2)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float=1e-5, inplace: bool=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma
        return x.mul_(gamma) if self.inplace else x * gamma


class GLU(nn.Module):
    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 channel_last: bool,
                 act_layer: Type[nn.Module],
                 bias: bool = True):
        super().__init__()
        # Different activation functions / versions of the gated linear unit:
        # - ReGLU:  Relu
        # - SwiGLU: Swish/SiLU
        # - GeGLU:  GELU
        # - GLU:    Sigmoid
        # seem to be the most promising once.
        # Extensive quantitative eval in table 1: https://arxiv.org/abs/2102.11972
        # Section 2 for explanation and implementation details: https://arxiv.org/abs/2002.05202
        # NOTE: Pytorch has a native GLU implementation: https://pytorch.org/docs/stable/generated/torch.nn.GLU.html?highlight=glu#torch.nn.GLU
        proj_out_dim = dim_out*2
        self.proj = nn.Linear(dim_in, proj_out_dim, bias=bias) if channel_last else \
            nn.Conv2d(dim_in, proj_out_dim, kernel_size=1, stride=1, bias=bias)
        self.channel_dim = -1 if channel_last else 1

        self.act_layer = act_layer

    def forward(self, x: torch.Tensor):
        x, gate = torch.tensor_split(self.proj(x), 2, dim=self.channel_dim)
        return x * self.act_layer(gate)


class MLP(nn.Module):
    def __init__(self,
                 dim: int,
                 channel_last: bool,
                 expansion_ratio: int,
                 act_layer: Type[nn.Module],
                 gated: bool = True,
                 bias: bool = True,
                 drop_prob: float = 0.):
        super().__init__()
        inner_dim = int(dim * expansion_ratio)
        if gated:
            # To keep the number of parameters (approx) constant regardless of whether glu == True
            # Section 2 for explanation: https://arxiv.org/abs/2002.05202
            #inner_dim = round(inner_dim * 2 / 3)
            #inner_dim = math.ceil(inner_dim * 2 / 3 / 32) * 32 # multiple of 32
            #inner_dim = round(inner_dim * 2 / 3 / 32) * 32 # multiple of 32
            inner_dim = math.floor(inner_dim * 2 / 3 / 32) * 32 # multiple of 32
            proj_in = GLU(dim_in=dim, dim_out=inner_dim, channel_last=channel_last, act_layer=act_layer, bias=bias)
        else:
            proj_in = nn.Sequential(
                nn.Linear(in_features=dim, out_features=inner_dim, bias=bias) if channel_last else \
                    nn.Conv2d(in_channels=dim, out_channels=inner_dim, kernel_size=1, stride=1, bias=bias),
                act_layer,
            )
        self.net = nn.Sequential(
            proj_in,
            nn.Dropout(p=drop_prob),
            nn.Linear(in_features=inner_dim, out_features=dim, bias=bias) if channel_last else \
                nn.Conv2d(in_channels=inner_dim, out_channels=dim, kernel_size=1, stride=1, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class PartitionAttentionCl(nn.Module):
    """ Grid or Block partition + Attn + FFN.
    NxC 'channels last' tensor layout.

    According to RW, NHWC attention is a few percent faster on GPUs (but slower on TPUs)
    https://github.com/rwightman/pytorch-image-models/blob/4f72bae43be26d9764a08d83b88f8bd4ec3dbe43/timm/models/maxxvit.py#L1258
    """

    def __init__(
            self,
            dim: int,
            partition_type: PartitionType,
            attention_cfg: dict,
            skip_first_norm: bool=False,
    ):
        super().__init__()
        norm_eps = attention_cfg.get('norm_eps', 1e-5)
        partition_size = attention_cfg.get("partition_size", (10, 3))
        use_torch_mha = attention_cfg.get("use_torch_mha", False)
        dim_head = attention_cfg.get('dim_head', 32)
        attention_bias = attention_cfg.get('attention_bias', True)
        mlp_act_string = attention_cfg.get("mlp_activation", "gelu")
        mlp_gated = attention_cfg.get("mlp_gated", False)
        mlp_bias = attention_cfg.get('mlp_bias', True)
        mlp_expand_ratio = attention_cfg.get('mlp_ratio', 4)

        drop_path = attention_cfg.get('drop_path', 0.0)
        drop_mlp = attention_cfg.get('drop_mlp', 0.0)
        ls_init_value = attention_cfg.get('ls_init_value', 1e-5)

        assert isinstance(use_torch_mha, bool)
        assert isinstance(mlp_gated, bool)
        assert_activation_string(activation_string=mlp_act_string)
        mlp_act_layer = nn.GELU()

        self_attn_module = TorchMHSAWrapperCl if use_torch_mha else SelfAttentionCl

        if isinstance(partition_size, int):
            partition_size = (partition_size, partition_size)
        else:
            partition_size = tuple(partition_size)
            assert len(partition_size) == 2
        self.partition_size = partition_size

        norm_layer = partial(nn.LayerNorm)  # NOTE this block is channels-last

        assert isinstance(partition_type, PartitionType)
        self.partition_window = partition_type == PartitionType.WINDOW

        self.norm1 = nn.Identity() if skip_first_norm else norm_layer(dim)
        self.self_attn = self_attn_module(dim,
                                          dim_head=dim_head,
                                          bias=attention_bias)
        self.ls1 = LayerScale(dim=dim, init_values=ls_init_value) if ls_init_value > 0 else nn.Identity()
        self.drop_path1 = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim = dim,
                       channel_last=True,
                       expansion_ratio = mlp_expand_ratio,
                       act_layer = mlp_act_layer,
                       gated = mlp_gated,
                       bias = mlp_bias,
                       drop_prob = drop_mlp)
        self.ls2 = LayerScale(dim=dim, init_values=ls_init_value) if ls_init_value > 0 else nn.Identity()
        self.drop_path2 = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()

    def _partition_attn(self, x):
        img_size = x.shape[1:3]
        if self.partition_window:
            partitioned = window_partition(x, self.partition_size)
        else:
            partitioned = grid_partition(x, self.partition_size)

        partitioned = self.self_attn(partitioned)

        if self.partition_window:
            x = window_reverse(partitioned, self.partition_size, (img_size[0], img_size[1]))
        else:
            x = grid_reverse(partitioned, self.partition_size, (img_size[0], img_size[1]))
        return x

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self._partition_attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


def window_partition(x, window_size: Tuple[int, int]):
    B, H, W, C = x.shape
    assert(H % window_size[0] == 0, f'height ({H}) must be divisible by window ({window_size[0]})')
    assert(W % window_size[1] == 0, f'width ({W}) must be divisible by window ({window_size[1]})')
    
    print(f'window_partition: B={B}, H={H}, W={W}, C={C}, window_size={window_size}')
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size: Tuple[int, int], img_size: Tuple[int, int]):
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x


def grid_partition(x, grid_size: Tuple[int, int]):
    B, H, W, C = x.shape
    assert(H % grid_size[0] == 0, f'height {H} must be divisible by grid {grid_size[0]}')
    assert(W % grid_size[1] == 0, f'width {W} must be divisible by grid {grid_size[1]}')
    x = x.view(B, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1], C)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, grid_size[0], grid_size[1], C)
    return windows


def grid_reverse(windows, grid_size: Tuple[int, int], img_size: Tuple[int, int]):
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(-1, H // grid_size[0], W // grid_size[1], grid_size[0], grid_size[1], C)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, H, W, C)
    return x


class TorchMHSAWrapperCl(nn.Module):
    """ Channels-last multi-head self-attention (B, ..., C) """
    def __init__(
            self,
            dim: int,
            dim_head: int = 32,
            bias: bool = True):
        super().__init__()
        assert dim % dim_head == 0
        num_heads = dim // dim_head
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, bias=bias, batch_first=True)

    def forward(self, x: torch.Tensor):
        restore_shape = x.shape
        B, C = restore_shape[0], restore_shape[-1]
        x = x.view(B, -1, C)
        attn_output, attn_output_weights =  self.mha(query=x, key=x, value=x)
        attn_output = attn_output.reshape(restore_shape)
        return attn_output


class SelfAttentionCl(nn.Module):
    """ Channels-last multi-head self-attention (B, ..., C) """
    def __init__(
            self,
            dim: int,
            dim_head: int = 32,
            bias: bool = True):
        super().__init__()
        self.num_heads = dim // dim_head
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        restore_shape = x.shape[:-1]

        q, k, v = self.qkv(x).view(B, -1, self.num_heads, self.dim_head * 3).transpose(1, 2).chunk(3, dim=3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(restore_shape + (-1,))
        x = self.proj(x)
        return x


def assert_activation_string(activation_string: Optional[Union[str, Tuple[str, ...], List[str]]]) -> None:
    # Serves as a hacky documentation and sanity check.
    # List of possible activation layer strings that are reasonable:
    # https://github.com/rwightman/pytorch-image-models/blob/a520da9b495422bc773fb5dfe10819acb8bd7c5c/timm/models/layers/create_act.py#L62
    if activation_string is None:
        return
    if isinstance(activation_string, str):
        assert activation_string in ('silu', 'swish', 'mish', 'relu', 'relu6', 'leaky_relu', 'elu', 'prelu', 'celu', 'selu',
                             'gelu', 'sigmoid', 'tanh', 'hard_sigmoid', 'hard_swish', 'hard_mish')
    elif isinstance(activation_string, (tuple, list)):
        for entry in activation_string:
            assert_activation_string(activation_string=entry)
    else:
        raise NotImplementedError


def assert_norm2d_layer_string(norm_layer: Optional[Union[str, Tuple[str, ...], List[str]]]) -> None:
    # Serves as a hacky documentation and sanity check.
    # List of possible norm layer strings that are reasonable:
    # https://github.com/rwightman/pytorch-image-models/blob/4f72bae43be26d9764a08d83b88f8bd4ec3dbe43/timm/models/layers/create_norm.py#L14
    if norm_layer is None:
        return
    if isinstance(norm_layer, str):
        assert norm_layer in ('batchnorm', 'batchnorm2d', 'groupnorm', 'layernorm2d')
    elif isinstance(norm_layer, (tuple, list)):
        for entry in norm_layer:
            assert_norm2d_layer_string(norm_layer=entry)
    else:
        raise NotImplementedError

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class MaxVitAttentionPairCl(nn.Module):
    def __init__(self,
                 dim: int,
                 skip_first_norm: bool,
                 attention_cfg: dict = {"use_torch_mha": False,
                                    "partition_size": (3, 10), # for 640 *192
                                    "dim_head": 32,
                                    "attention_bias": True,
                                    "mlp_activation": "gelu",
                                    "mlp_gated": False,
                                    "mlp_bias": True,
                                    "mlp_ratio": 4,
                                    "drop_mlp": 0,
                                    "drop_path": 0,
                                    "ls_init_value": 1e-5}):
        super().__init__()

        self.att_window = PartitionAttentionCl(dim=dim,
                                               partition_type=PartitionType.WINDOW,
                                               attention_cfg=attention_cfg,
                                               skip_first_norm=skip_first_norm)
        self.att_grid = PartitionAttentionCl(dim=dim,
                                             partition_type=PartitionType.GRID,
                                             attention_cfg=attention_cfg,
                                             skip_first_norm=False)

    def forward(self, x):
        # reshape to channel-last inputs:
        x = x.permute((0, 2, 3, 1))
        x = self.att_window(x)
        x = self.att_grid(x)
        x = x.permute((0, 3, 1, 2))
        return x
