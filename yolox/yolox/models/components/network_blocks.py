#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

#-----------------------------------------------------------------
from yolox.nv_qdq import QDQ
#QDQ.quant_nn
#-----------------------------------------------------------------

class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu", 
        quantize: bool = False, quantize_weight_only: bool = False):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        if quantize:
          if quantize_weight_only:####lixxxx
            self.conv = QDQ.quant_nn.QuantConv2d_WeightOnly(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
          else:
            self.conv = QDQ.quant_nn.QuantConv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        else:
          self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu", quantize: bool = False):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
            quantize=quantize 
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act, quantize=quantize  
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
            self,
            in_channels,
            out_channels,
            shortcut=True,
            expansion=0.5,
            depthwise=False,
            act="silu",
            quantize: bool = False
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act, quantize=quantize, quantize_weight_only=True)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act, quantize=quantize)
        self.use_add = shortcut and in_channels == out_channels
        self._quantize = quantize
        if self._quantize: 
            self.residual_quantizer = QDQ.quant_nn.TensorQuantizer(QDQ.quant_nn.QuantConv2d.default_quant_desc_input)

    def forward(self, x):
      if self._quantize: 
        x = self.residual_quantizer(x)
        identity = x
        y = self.residual_quantizer(self.conv2(self.conv1(x)))
        if self.use_add: #-------------------------------
            y += identity
        return y    
      else:
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int, quantize: bool = False):  
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, act="lrelu", quantize=quantize
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, act="lrelu", quantize=quantize
        )

        self._quantize = quantize             
        if self._quantize:  
            self.residual_quantizer = QDQ.quant_nn.TensorQuantizer(QDQ.quant_nn.QuantConv2d.default_quant_desc_input)

    def forward(self, x):
        #identity = x
        out = self.layer2(self.layer1(x))

        if self._quantize:  
            out += self.residual_quantizer(x) # identity
        else:
            out += x # identity

        return out


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu", 
		    quantize:bool = False):
        super().__init__()
        hidden_channels = in_channels // 2
        self._quantize = quantize
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation, quantize=quantize)
        # self.m = nn.ModuleList(
        #     [
        #         nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
        #         for ks in kernel_sizes
        #     ]
        # )
        list = []
        for ks in kernel_sizes:
            if ks == 5:
              if self._quantize:
                list.append(QDQ.quant_nn.QuantMaxPool2d(kernel_size=ks, stride=1, padding=ks // 2))
              else:
                list.append(nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2))
 
            elif ks == 9:
              if self._quantize:
                list.append(QDQ.quant_nn.QuantMaxPool2d(kernel_size=7, stride=1, padding=3))
              else:
                list.append(nn.MaxPool2d(kernel_size=7, stride=1, padding=3))  
            else:
              if self._quantize:
                list.append(QDQ.quant_nn.QuantMaxPool2d(kernel_size=3, stride=1, padding=1))
              else:
                list.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
  

        self.m = nn.ModuleList(list)
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation, quantize=quantize)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
            self,
            in_channels,
            out_channels,
            n=1,
            shortcut=True,
            expansion=0.5,
            depthwise=False,
            act="silu",
            quantize:bool = False,
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act, quantize=quantize) # ----- Concat
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act, quantize=quantize)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act, quantize=quantize)

        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act, quantize=quantize
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu", quantize: bool = False):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act, quantize=quantize)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1, )
        return self.conv(x)


class Focus2(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu", quantize: bool = False):
        super().__init__()
        self.conv0 = BaseConv(in_channels, in_channels * 4, ksize, stride, act=act, quantize=quantize)
        self.conv1 = BaseConv(in_channels * 4, out_channels // 4, ksize, stride=2, act=act, quantize=quantize)
        self.conv2 = BaseConv(out_channels, out_channels, ksize, stride, act=act, quantize=quantize)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        x = self.conv0(x)
        patch_top_left = self.conv1(x)
        patch_top_right = self.conv1(x)
        patch_bot_left = self.conv1(x)
        patch_bot_right = self.conv1(x)
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1, )
        return self.conv2(x)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, k, s, padding, dilation, act='silu', 
        quantize: bool = False):
        activate = get_activation(act, inplace=True)

        if quantize:
          modules = [QDQ.quant_nn.QuantConv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=padding, dilation=dilation, bias=False),
                    nn.BatchNorm2d(out_channels),
                    activate]
        else:
          modules = [nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=padding, dilation=dilation, bias=False),
                    nn.BatchNorm2d(out_channels),
                    activate]
        super(ASPPConv, self).__init__(*modules)


class ASPP(nn.Module):
    def __init__(self, input=3, depth=32, act='silu', quantize: bool = False):
        super(ASPP, self).__init__()
        activate = get_activation(act, inplace=True)
        
        if quantize:  
          self.conv = nn.Sequential(QDQ.quant_nn.QuantConv2d(input, 32, 3, 2, 1),
                                    nn.BatchNorm2d(32),
                                    activate)
        else:
          self.conv = nn.Sequential(nn.Conv2d(input, 32, 3, 2, 1),
                          nn.BatchNorm2d(32),
                          activate)
        in_channel = 32
        self.atrous_block1 = ASPPConv(in_channel, depth, 1, 1, 0, 1, act=act, quantize=quantize)
        self.atrous_block6 = ASPPConv(in_channel, depth, 5, 1, padding=4, dilation=2, act=act, quantize=quantize)
        # self.atrous_block12 = ASPPConv(in_channel, depth, 3, 1, padding=12, dilation=12)
        # self.atrous_block18 = ASPPConv(in_channel, depth, 3, 1, padding=18, dilation=18)
        if quantize:  
          self.channel_reduce = nn.Sequential(QDQ.quant_nn.QuantConv2d(64, 32, 3, 1, 1),
                                              nn.BatchNorm2d(32),
                                              activate)
        else:
          self.channel_reduce = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1),
                                    nn.BatchNorm2d(32),
                                    activate)
    def forward(self, x):
        x = self.conv(x)
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        # atrous_block12 = self.atrous_block12(x)
        # atrous_block18 = self.atrous_block18(x)

        net = torch.cat([atrous_block1, atrous_block6], dim=1)
        out = self.channel_reduce(net)

        return out
