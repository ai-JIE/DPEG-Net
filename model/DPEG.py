import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
# from fvcore.nn import FlopCountAnalysis
from ptflops import get_model_complexity_info

from segformer.segformer import *

from torch import nn
import torch
import torch.nn.functional as F


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class group_aggregation_bridge(nn.Module):
    def __init__(self, dim_xh, dim_xl, k_size=3, d_list=[1, 2, 5, 7]):
        super().__init__()
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, kernel_size=(1, 1))
        group_size = dim_xl // 2

        # 空间注意力
        self.g0 = nn.Sequential(
            LayerNorm(normalized_shape=group_size, data_format='channels_first'),
            nn.Conv2d(group_size, group_size, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[0] - 1)) // 2,
                      dilation=d_list[0], groups=group_size)
        )  # padding=1
        self.g1 = nn.Sequential(
            LayerNorm(normalized_shape=group_size, data_format='channels_first'),
            nn.Conv2d(group_size, group_size, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[1] - 1)) // 2,
                      dilation=d_list[1], groups=group_size)
        )  # padding=2
        self.g2 = nn.Sequential(
            LayerNorm(normalized_shape=group_size, data_format='channels_first'),
            nn.Conv2d(group_size, group_size, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[2] - 1)) // 2,
                      dilation=d_list[2], groups=group_size)
        )  # padding=5
        self.g3 = nn.Sequential(
            LayerNorm(normalized_shape=group_size, data_format='channels_first'),
            nn.Conv2d(group_size, group_size, kernel_size=3, stride=(1, 1),
                      padding=(k_size + (k_size - 1) * (d_list[3] - 1)) // 2,
                      dilation=d_list[3], groups=group_size)
        )
        self.tail_conv = nn.Sequential(
            LayerNorm(normalized_shape=dim_xl * 2, data_format='channels_first'),
            nn.Conv2d(dim_xl * 2, dim_xl, 1)
        )
        self.conv = nn.Conv2d(dim_xl * 2, dim_xl, kernel_size=(1, 1))
        self.act = nn.GELU()

        # 通道注意力
        self.channel_attn = nn.Sequential(
            nn.Conv2d(dim_xl, dim_xl // 2, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_xl // 2, dim_xl, kernel_size=(1, 1)),
            nn.Sigmoid()
        )

        # 门控注意力加在空间注意力上的
        self.psi = nn.Sequential(
            nn.Conv2d(dim_xl, dim_xl, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True),
            nn.BatchNorm2d(dim_xl),
            nn.Sigmoid()
        )

        # 残差连接
        self.residual = nn.Sequential(

            nn.BatchNorm2d(dim_xl),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_xl, dim_xl // 2, kernel_size=(1, 1)),
            nn.BatchNorm2d(int(dim_xl // 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_xl // 2, dim_xl // 2, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(int(dim_xl // 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_xl // 2, dim_xl, kernel_size=(1, 1))
        )


    def forward(self, xH, xL):
        xH = self.pre_project(xH)
        # xh作为下层特征，所以需要下行代码上采样(空间注意力)
        xH = F.interpolate(xH, size=[xL.size(2), xL.size(3)], mode='bilinear', align_corners=True)
        xh = torch.chunk(xH, 4, dim=1)
        xl = torch.chunk(xL, 4, dim=1)
        x0 = self.g0(torch.cat((xh[0], xl[0]), dim=1))
        x0 = self.act(x0)
        x1 = self.g1(torch.cat((xh[1], xl[1]), dim=1))
        x1 = self.act(x1)
        x2 = self.g2(torch.cat((xh[2], xl[2]), dim=1))
        x2 = self.act(x2)
        x3 = self.g3(torch.cat((xh[3], xl[3]), dim=1))
        x3 = self.act(x3)

        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.act(self.tail_conv(x))

        residual = torch.cat([xH, xL], dim=1)
        residual = self.act(self.conv(residual))

        # 通道注意力
        channel_x = self.channel_attn(residual)

        x = residual + x
        x = self.act(x)
        # print(x.shape)
        psi = self.psi(x)
        spatial_attn = psi * x

        attn = torch.cat([channel_x, spatial_attn], dim=1)
        attn = self.act(self.conv(attn))
        # attn = self._7conv(attn)
        x = self.residual(attn)
        x = x + attn

        # 调制形状适配Dual-tansformer
        b, c, _, _ = x.shape
        x = x.permute(0, 2, 3, 1).view(b, -1, c)

        return x



class SqueezeExcite(nn.Module):
    """ Squeeze-and-Excitation w/ specific features for EfficientNet/MobileNet family

    Args:
        in_chs (int): input channels to layer
        rd_ratio (float): ratio of squeeze reduction
        act_layer (nn.Module): activation layer of containing block
        gate_layer (Callable): attention gate function
        force_act_layer (nn.Module): override block's activation fn if this is set/bound
        rd_round_fn (Callable): specify a fn to calculate rounding of reduced chs
    """

    def __init__(
            self, in_chs, rd_ratio=0.25, rd_channels=None, rd_round_fn=None):
        super(SqueezeExcite, self).__init__()
        if rd_channels is None:
            rd_round_fn = rd_round_fn or round
            rd_channels = rd_round_fn(in_chs * rd_ratio)
        self.conv_reduce = nn.Conv2d(in_chs, rd_channels, 1, bias=True)
        self.act1 = nn.GELU()
        self.conv_expand = nn.Conv2d(rd_channels, in_chs, 1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


#
# class EfficientAttention(nn.Module):
#     """
#     input  -> x:[B, D, H, W]
#     output ->   [B, D, H, W]
#
#     in_channels:    int -> Embedding Dimension
#     key_channels:   int -> Key Embedding Dimension,   Best: (in_channels)
#     value_channels: int -> Value Embedding Dimension, Best: (in_channels or in_channels//2)
#     head_count:     int -> It divides the embedding dimension by the head_count and process each part individually
#
#     Conv2D # of Params:  ((k_h * k_w * C_in) + 1) * C_out)
#     """
#
#     def __init__(self, in_channels, key_channels, value_channels, head_count=1):
#         super().__init__()
#         self.in_channels = in_channels
#         self.key_channels = key_channels
#         self.head_count = head_count
#
#         self.value_channels = value_channels
#
#         self.keys = nn.Conv2d(in_channels, key_channels, 1)
#         self.queries = nn.Conv2d(in_channels, key_channels, 1)
#         # self.qk = nn.Linear(in_channels, in_channels * 2)
#         self.values = nn.Conv2d(in_channels, value_channels, 1)
#         self.reprojection = nn.Conv2d(value_channels, in_channels, 1)
#         self.temperature = nn.Parameter(torch.ones(value_channels, 1, 1))
#
#
#     def forward(self, input_, qk):
#         n, _, h, w = input_.size()
#
#         # 改动的地方
#         # keys = self.keys(input_).reshape((n, self.key_channels, h, w))
#         # queries = self.queries(input_).reshape(n, self.key_channels, h, w)
#         # values = self.values(input_).reshape((n, self.value_channels, h, w))
#
#         keys = self.keys(qk).reshape((n, self.key_channels, h * w))
#         queries = self.queries(qk).reshape(n, self.key_channels, h * w)
#         values = self.values(input_).reshape((n, self.value_channels, h * w))
#
#         head_key_channels = self.key_channels // self.head_count
#         head_value_channels = self.value_channels // self.head_count
#
#         attended_values = []
#         for i in range(self.head_count):
#             key = F.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)
#
#             query = F.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)
#
#             value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]
#
#             # 改动的地方
#             context = key * value
#
#             attended_value = (context * query* self.temperature).reshape(n, head_value_channels, h, w)  # n*dv
#             attended_values.append(attended_value)
#             # context = key @ value.transpose(1, 2)  # dk*dv
#             # attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)  # n*dv
#             # attended_values.append(attended_value)
#
#         aggregated_values = torch.cat(attended_values, dim=1)
#         attention = self.reprojection(aggregated_values)
#
#         return attention


class EfficientAttention(nn.Module):
    """
    input  -> x:[B, D, H, W]
    output ->   [B, D, H, W]

    in_channels:    int -> Embedding Dimension
    key_channels:   int -> Key Embedding Dimension,   Best: (in_channels)
    value_channels: int -> Value Embedding Dimension, Best: (in_channels or in_channels//2)
    head_count:     int -> It divides the embedding dimension by the head_count and process each part individually

    Conv2D # of Params:  ((k_h * k_w * C_in) + 1) * C_out)
    """

    def __init__(self, in_channels, key_channels, value_channels, head_count=1):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count

        self.value_channels = value_channels
        self.attn_drop = nn.Dropout(0.0)
        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)
        self.temperature = nn.Parameter(torch.ones(value_channels, 1, 1))
        self.proj_drop = nn.Dropout(0.0)



    def forward(self, input_):
        n, _, h, w = input_.size()

        # 改动的地方
        keys = self.keys(input_).reshape((n, self.key_channels, h, w))
        queries = self.queries(input_).reshape(n, self.key_channels, h, w)
        values = self.values(input_).reshape((n, self.value_channels, h, w))

        # keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        # queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        # values = self.values(input_).reshape((n, self.value_channels, h * w))

        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)
            query = F.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]
            context = key * value
            attended_value = (context * query * self.temperature).reshape(n, head_value_channels, h, w)  # n*dv
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)





        return attention


class ChannelAttention(nn.Module):
    """
    Input -> x: [B, N, C]
    Output -> [B, N, C]
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0, proj_drop=0):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """x: [B, N, C]"""
        B, N, C = x.shape
        # 修改的地方

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # qkv = qkv.permute(2, 0, 3, 1, 4)
        # q, k = qkv[0], qkv[1]

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        # v = x.transpose(-2, -1).reshape(B,self.num_heads,C // self.num_heads,N)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # attn = (q @ k.transpose(-2, -1)) * self.temperature

        # 改动的地方
        attn = (q * k) * self.temperature

        # -------------------
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 改动的地方

        x = (attn * v).permute(0, 3, 1, 2).reshape(B, N, C)

        # x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        # ------------------
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class DualTransformerBlock(nn.Module):
    """
    Input  -> x (Size: (b, (H*W), d)), H, W
    Output -> (b, (H*W), d)
    """

    # stage0 是（128，56，56）  stage1 是（320，28，28） stage0 是（512，14，14）
    def __init__(self, in_dim, key_dim, value_dim, head_count=1, token_mlp="mix"):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = EfficientAttention(in_channels=in_dim, key_channels=key_dim, value_channels=value_dim, head_count=1)
        self.norm2 = nn.LayerNorm(in_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.channel_attn = ChannelAttention(in_dim)
        self.norm4 = nn.LayerNorm(in_dim)
        self.norm5 = nn.LayerNorm(in_dim)
        self.norm6 = nn.LayerNorm(in_dim)
        # self.norm7 = nn.LayerNorm(in_dim)
        # self.norm8 = nn.LayerNorm(in_dim)
        self.act1 = nn.ReLU()
        # self.act2 = nn.ReLU()
        self.act3 = nn.ReLU()

        self.conv1 = nn.Conv2d(in_dim * 2, in_dim, kernel_size=(1, 1))
        self.conv3 = nn.Conv2d(in_dim, in_dim, kernel_size=(3, 3), stride=1, padding=1)
        # self.conv3 = nn.Conv2d(in_dim, in_dim, kernel_size=(3, 3), stride=1, padding=1, groups=in_dim)

        # se模块
        self.SE = SqueezeExcite(in_dim)
        # 合并模块
        self.residual = nn.Sequential(

            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim // 2, kernel_size=(1, 1)),
            nn.BatchNorm2d(int(in_dim // 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim // 2, in_dim // 2, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(int(in_dim // 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim // 2, in_dim, kernel_size=(1, 1))
        )
        if token_mlp == "mix":
            self.mlp1 = MixFFN(in_dim, int(in_dim * 4))
            self.mlp2 = MixFFN(in_dim, int(in_dim * 4))
        elif token_mlp == "mix_skip":
            self.mlp1 = MixFFN_skip(in_dim, int(in_dim * 4))
            self.mlp2 = MixFFN_skip(in_dim, int(in_dim * 4))
        else:
            self.mlp1 = MLP_FFN(in_dim, int(in_dim * 4))
            self.mlp2 = MLP_FFN(in_dim, int(in_dim * 4))

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:

        # 双路径
        # norm1 = self.norm1(x)
        # norm1 = Rearrange("b (h w) d -> b d h w", h=H, w=W)(norm1)
        # norm1 = self.conv3(norm1)
        # norm1 = self.act3(norm1)

        # 全局语义路径
        seg = self.norm5(x)
        channel_attn = self.channel_attn(seg)
        add3 = seg + channel_attn
        norm4 = self.norm4(add3)

        norm4 = Rearrange("b (h w) d -> b d h w", h=H, w=W)(norm4)

        se = self.SE(norm4)
        se = norm4 + se
        se = Rearrange("b d h w -> b (h w) d", h=H, w=W)(se)
        se = self.norm6(se)
        mlp2 = se + self.mlp2(se, H, W)


        # 像素级路径
        norm1 = self.norm1(x)
        norm1 = Rearrange("b (h w) d -> b d h w", h=H, w=W)(norm1)
        norm1 = self.conv3(norm1)
        norm1 = self.act3(norm1)
        # # 由语义路径过来的qk
        attn = self.attn(norm1)
        attn = Rearrange("b d h w -> b (h w) d")(attn)

        add1 = x + attn
        norm2 = self.norm2(add1)
        mlp1 = self.mlp1(norm2, H, W)
        add2 = add1 + mlp1
        # # 双路径transformer之后的合并模块1（相加的）
        # # norm3 = self.norm3(add2 + mlp2)
        # # norm3 = Rearrange("b (h w) d -> b d h w", h=H, w=W)(norm3)
        # # mx = norm3 + self.residual(norm3)
        # # mx = Rearrange("b d h w -> b (h w) d")(mx)

        # 双路径transformer之后的合并模块1（concat）
        merge = torch.cat((add2, mlp2), dim=2)
        merge = Rearrange("b (h w) d -> b d h w", h=H, w=W)(merge)
        merge = self.act1(self.conv1(merge))
        mx = merge + self.residual(merge)
        mx = Rearrange("b d h w -> b (h w) d")(mx)



        return mx


# Encoder
class MiT(nn.Module):
    def __init__(self, image_size, in_dim, key_dim, value_dim, layers, head_count=1, token_mlp="mix_skip"):
        super().__init__()
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        padding_sizes = [3, 1, 1, 1]

        # patch_embed
        # layers = [2, 2, 2, 2] dims = [64, 128, 320, 512]
        self.patch_embed1 = OverlapPatchEmbeddings(
            image_size, patch_sizes[0], strides[0], padding_sizes[0], 3, in_dim[0]
        )
        self.patch_embed2 = OverlapPatchEmbeddings(
            image_size // 4, patch_sizes[1], strides[1], padding_sizes[1], in_dim[0], in_dim[1]
        )
        self.patch_embed3 = OverlapPatchEmbeddings(
            image_size // 8, patch_sizes[2], strides[2], padding_sizes[2], in_dim[1], in_dim[2]
        )

        # transformer encoder
        self.block1 = nn.ModuleList(
            [DualTransformerBlock(in_dim[0], key_dim[0], value_dim[0], head_count, token_mlp) for _ in range(layers[0])]
        )
        self.norm1 = nn.LayerNorm(in_dim[0])

        self.block2 = nn.ModuleList(
            [DualTransformerBlock(in_dim[1], key_dim[1], value_dim[1], head_count, token_mlp) for _ in range(layers[1])]
        )
        self.norm2 = nn.LayerNorm(in_dim[1])

        self.block3 = nn.ModuleList(
            [DualTransformerBlock(in_dim[2], key_dim[2], value_dim[2], head_count, token_mlp) for _ in range(layers[2])]
        )
        self.norm3 = nn.LayerNorm(in_dim[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs


# Decoder
class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # print("x_shape-----",x.shape)
        H, W = self.input_resolution
        x = self.expand(x)

        B, L, C = x.shape
        # print(x.shape)
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x.clone())

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(
            x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale ** 2)
        )
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x.clone())

        return x


class MyDecoderLayer(nn.Module):
    def __init__(
            self, input_size, in_out_chan, head_count, token_mlp_mode, n_class=9, norm_layer=nn.LayerNorm, is_last=False
    ):
        super().__init__()
        dims = in_out_chan[5]
        out_dim = in_out_chan[1]
        key_dim = in_out_chan[2]
        value_dim = in_out_chan[3]
        if not is_last:
            self.cross_attn = group_aggregation_bridge(dims, out_dim)
            # transformer decoder
            self.last_layer = None

        else:
            self.cross_attn = group_aggregation_bridge(dims, out_dim)
            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

            # transformer decoder

        self.layer_former_1 = DualTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)
        self.layer_former_2 = DualTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1, x2):
        # 修改的加入新的交叉注意力模块
        b, c, h, w = x2.shape
        cat_linear_x = self.cross_attn(x1, x2)
        tran_layer_1 = self.layer_former_1(cat_linear_x, h, w)
        tran_layer_2 = self.layer_former_2(tran_layer_1, h, w)
        if self.last_layer:
            out = self.last_layer(self.layer_up(tran_layer_2).view(b, 4 * h, 4 * w, -1).permute(0, 3, 1, 2))
        else:
            # out = self.layer_up(tran_layer_2)
            b, l, c = tran_layer_2.shape
            out = tran_layer_2.permute(0, 2, 1).view(b, c, h, w)  # 8,320,28,28

        return out


class DAEFormer(nn.Module):
    def __init__(self, num_classes=9, head_count=1, token_mlp_mode="mix_skip"):
        super().__init__()

        # Encoder
        dims, key_dim, value_dim, layers = [[128, 320, 512], [128, 320, 512], [128, 320, 512], [2, 2, 2]]
        self.backbone = MiT(
            image_size=224,
            in_dim=dims,
            key_dim=key_dim,
            value_dim=value_dim,
            layers=layers,
            head_count=head_count,
            token_mlp=token_mlp_mode,
        )

        # Decoder
        d_base_feat_size = 7  # 16 for 512 input size, and 7 for 224
        in_out_chan = [
            [64, 128, 128, 128, 160, 320],
            [320, 320, 320, 320, 256, 512],
            [512, 512, 512, 512, 512],
        ]  # [dim, out_dim, key_dim, value_dim, x2_dim]

        self.decoder_1 = MyDecoderLayer(
            (d_base_feat_size * 4, d_base_feat_size * 4),
            in_out_chan[1],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
        )
        self.decoder_0 = MyDecoderLayer(
            (d_base_feat_size * 8, d_base_feat_size * 8),
            in_out_chan[0],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
            is_last=True,
        )

    def forward(self, x):
        # ---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # backbone  是  MIT
        output_enc = self.backbone(x)

        b, c, _, _ = output_enc[2].shape

        # 改后---------------Decoder-------------------------
        tmp_1 = self.decoder_1(output_enc[2], output_enc[1])
        tmp_0 = self.decoder_0(tmp_1, output_enc[0])

        return tmp_0


if __name__ == '__main__':
    image = torch.randn(1, 3, 224, 224)
    model = DAEFormer(num_classes=9)
    out = model(image)
    print(out.shape)
    macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

"""

最原始的三个创新点
Computational complexity:       32.58 GMac
Number of parameters:           42.64 M 

"""