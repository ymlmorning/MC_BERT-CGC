
from torch import nn
import torch


class LayerNorm(nn.Module):
    def __init__(self, shape=(1, 7, 1, 1), dim_index=1):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))
        self.dim_index = dim_index
        self.eps = 1e-6

    def forward(self, x):
        """

        :param x: bsz x dim x max_len x max_len
        :param mask: bsz x dim x max_len x max_len, 为1的地方为pad
        :return:
        """
        u = x.mean(dim=self.dim_index, keepdim=True)
        s = (x - u).pow(2).mean(dim=self.dim_index, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight * x + self.bias
        return x


import torch.nn.functional as F
class OctConv2d_v1(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 alpha_in=0.5,
                 alpha_out=0.5
                 ):
        """adapt first octconv , octconv and last octconv

        """
        assert alpha_in >= 0 and alpha_in <= 1, "the value of alpha_in should be in range of [0,1],but get {}".format(
            alpha_in)
        assert alpha_out >= 0 and alpha_out <= 1, "the value of alpha_in should be in range of [0,1],but get {}".format(
            alpha_out)
        super(OctConv2d_v1, self).__init__(in_channels,
                                        out_channels,
                                        dilation,
                                        groups,
                                        bias,)
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.kernel_size = (1,1)
        self.stride = (1,1)
        self.avgPool = nn.AvgPool2d(kernel_size, stride, padding)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.inChannelSplitIndex = int(
            self.alpha_in * self.in_channels)
        self.outChannelSplitIndex = int(
            self.alpha_out * self.out_channels)
        # split bias
        if bias:
            self.hh_bias = self.bias[self.outChannelSplitIndex:]
            self.hl_bias = self.bias[:self.outChannelSplitIndex]
            self.ll_bias = self.bias[ :self.outChannelSplitIndex]
            self.lh_bias = self.bias[ self.outChannelSplitIndex:]
        else:
            self.hh_bias = None
            self.hl_bias = None
            self.ll_bias = None
            self.ll_bias = None

        # conv and upsample
        self.upsample = F.interpolate

    def forward(self, x):
        if not isinstance(x, tuple):
            # first octconv
            input_h = x if self.alpha_in == 0 else None
            input_l = x if self.alpha_in == 1 else None
        else:
            input_l = x[0]
            input_h = x[1]

        output = [0, 0]
        # H->H
        if self.outChannelSplitIndex != self.out_channels and self.inChannelSplitIndex != self.in_channels:
            output_hh = F.conv2d(self.avgPool(input_h),
                                 self.weight[
                                 self.outChannelSplitIndex:,
                                 self.inChannelSplitIndex:,
                                 :, :],
                                 self.bias[self.outChannelSplitIndex:],
                                 self.kernel_size
                                 )

            output[1] += output_hh

        # H->L
        if self.outChannelSplitIndex != 0 and self.inChannelSplitIndex != self.in_channels:
            output_hl = F.conv2d(self.avgpool(self.avgPool(input_h)),
                                 self.weight[
                :self.outChannelSplitIndex,
                self.inChannelSplitIndex:,
                                     :, :],
                                 self.bias[:self.outChannelSplitIndex],
                                 self.kernel_size
                                 )

            output[0] += output_hl

        # L->L
        if self.outChannelSplitIndex != 0 and self.inChannelSplitIndex != 0:
            output_ll = F.conv2d((self.avgPool(input_l)),
                                 self.weight[
                                 :self.outChannelSplitIndex,
                                 :self.inChannelSplitIndex,
                                 :, :],
                                 self.bias[:self.outChannelSplitIndex],
                                 self.kernel_size
                                 )

            output[0] += output_ll

        # L->H
        if self.outChannelSplitIndex != self.out_channels and self.inChannelSplitIndex != 0:
            output_lh = F.conv2d(self.avgPool(input_l),
                                 self.weight[
                                 self.outChannelSplitIndex:,
                                 :self.inChannelSplitIndex,
                                 :, :],
                                 self.bias[self.outChannelSplitIndex:],
                                 self.kernel_size
                                 )
            output_lh = self.upsample(output_lh, scale_factor=2)

            output[1] += output_lh

        if isinstance(output[0], int):
            out = output[1]
        else:
            out = tuple(output)
        return out

class MaskConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, groups=1):
        super(MaskConv2d, self).__init__()
        #self.conv2d = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding,bias=False, groups=groups)
        self.conv2d = OctConv2d_v1(in_ch, out_ch, kernel_size=kernel_size, padding=padding,bias=False, groups=groups)

    def forward(self, x, mask):
        """

        :param x:
        :param mask:
        :return:
        """
        x = x.masked_fill(mask, 0)
        _x = self.conv2d(x)
        return _x


class MaskCNN(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, depth=3):
        super(MaskCNN, self).__init__()

        layers = []
        for i in range(depth):
            layers.extend([
                MaskConv2d(input_channels, input_channels, kernel_size=kernel_size, padding=kernel_size//2),
                LayerNorm((1, input_channels, 1, 1), dim_index=1),
                nn.GELU()])
        layers.append(MaskConv2d(input_channels, output_channels, kernel_size=3, padding=3//2))
        self.cnns = nn.ModuleList(layers)

    def forward(self, x, mask):
        _x = x  # 用作residual
        for layer in self.cnns:
            if isinstance(layer, LayerNorm):
                x = x + _x
                x = layer(x)
                _x = x
            elif not isinstance(layer, nn.GELU):
                x = layer(x, mask)
            else:
                x = layer(x)
        return _x