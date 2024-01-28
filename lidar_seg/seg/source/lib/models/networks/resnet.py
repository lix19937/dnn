# Copyright.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from torch.nn import functional as F
from models.decode import lidar_od_decode, lidar_post_gpu
from models.utils import _sigmoid


class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(
            pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn = self.bn(conv)
        relu = self.relu(bn)
        return relu


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet34(nn.Module):
    def __init__(self, heads, phase='train'):
        super(ResNet34, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(16, 128, 7, stride=2, padding=3,
                      bias=False),  # downsample 2
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)  # k=3, s=2, p=1. downsample 2
        )

        self.layer1 = self.make_layer(128, 128, 3)
        self.layer2 = self.make_layer(128, 256, 4, stride=1)  # No downsample.

        self.heads = heads
        self.phase = phase

        # keypoint heatmaps
        for head in heads.keys():
            if 'hm' in head:
                module = nn.ModuleList([
                    self.make_kp_layer(
                        256, 256, heads[head])
                ])
                self.__setattr__(head, module)
                for heat in self.__getattr__(head):
                    heat[-1].bias.data.fill_(-2.19)
            else:
                module = nn.ModuleList([
                    self.make_kp_layer(
                        256, 256, heads[head])
                ])
                self.__setattr__(head, module)

    def make_kp_layer(self, cnv_dim, curr_dim, out_dim):
        return nn.Sequential(
            convolution(3, cnv_dim, curr_dim, with_bn=False),
            nn.Conv2d(curr_dim, out_dim, (1, 1))
        )

    def make_layer(self, in_ch, out_ch, block_num, stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        layers = []
        layers.append(ResidualBlock(in_ch, out_ch, stride, shortcut))

        for _ in range(1, block_num):
            layers.append(ResidualBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        outs = []
        out = {}
        for head in self.heads:
            layer = self.__getattr__(head)[0]
            y = layer(x)
            out[head] = y
        outs.append(out)
        # Decoding boxes in gpu when doing infer.
        if self.phase != 'train':
            infer_res = outs[-1]
            hm = _sigmoid(infer_res['hm'].detach())
            rot = infer_res['rot'].detach()
            z = infer_res['z'].detach()
            dim = infer_res['dim'].detach()
            reg = infer_res['reg'].detach()
            top_k = 128
            # B K 10 [xs, ys, z, dim, rot, scores, clses].
            dets = lidar_od_decode(hm, rot, z, dim, reg, top_k)
            # K 9: x,y,z,l,w,h,theta,score,cls
            dets_post = lidar_post_gpu(dets[0, :, :])
            outs[-1].update({'decoded': dets_post})
        return outs


def get_resnet34(heads, head_conv, num_stacks, phase='train'):
    model = ResNet34(heads, phase)
    return model
