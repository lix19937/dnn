# Copyright.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.decode import lidar_od_decode, lidar_post_gpu
from models.utils import _sigmoid


class SingleConv(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(SingleConv, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, k, padding=pad,
        stride=(stride, stride), bias=not with_bn)
        self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SingleConv1D(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super().__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv1d(inp_dim, out_dim, k, padding=pad, stride=stride, bias=not with_bn)
        self.bn = nn.BatchNorm1d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Residual, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), padding=(
            1, 1), stride=(stride, stride), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3),
                               padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1), stride=(
                stride, stride), bias=False),
            nn.BatchNorm2d(out_channels)
        ) if stride != 1 or in_channels != out_channels else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.relu1(res)

        res = self.conv2(res)
        res = self.bn2(res)

        skip = self.skip(x)
        return self.relu(res + skip)


class Stem(nn.Module):
    """7*7 conv + residual. Downsample ratio = 4"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.stem = nn.Sequential(
            SingleConv(7, in_channels, out_channels//2, stride=2),
            Residual(out_channels//2, out_channels, stride=2)
        )

    def forward(self, x):
        return self.stem(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, scale_factor=2):
        super().__init__()

        # If bilinear, use the normal convolutions to reduce the number of channels.
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=scale_factor, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels // 2, in_channels // scale_factor, kernel_size=scale_factor, stride=scale_factor)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is NCHW.
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        # pad = nn.ReplicationPad2d(padding=(diffX // 2, diffX - diffX // 2,
        #                                    diffY // 2, diffY - diffY // 2))
        # x1 = pad(x1)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SegHeader(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        tmp_channels = 64
        self.convs = SingleConv(3, in_channels, tmp_channels)
        # self.conv_pt = nn.Conv1d(3, tmp_channels, kernel_size=1)
        self.conv_pt = SingleConv1D(1, 3, tmp_channels)
        self.conv_seg = nn.Conv1d(tmp_channels * 2, num_classes, kernel_size=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, indices, mask, aligned_cloud):
        """Forward function."""
        output = self.convs(x)

        shape = output.shape
        pt_fea = self.conv_pt(aligned_cloud)
        tmp = indices * mask
        tmp = tmp.expand(shape[0], shape[1], tmp.shape[-1])
        reshaped_input = output.reshape(shape[0], shape[1], -1)
        packed_cloud = reshaped_input.gather(2, tmp.long())
        packed_cloud = torch.cat((pt_fea, packed_cloud), dim=1)

        feat = self.dropout(packed_cloud)
        output = self.conv_seg(feat)
        return output

class UNet(nn.Module):
    def __init__(self, seg_class_num, phase, bilinear=True):
        super(UNet, self).__init__()
        n_channels = 16
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.phase = phase

        self.pre = Stem(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.seg_up = Up(80, 32, bilinear, scale_factor=4)
        self.seg = SegHeader(32, seg_class_num)

    def forward(self, x):
        # Unet backbone.
        x1 = self.pre(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x4 = self.up1(x5, x4)
        x3 = self.up2(x4, x3)
        x2 = self.up3(x3, x2)
        x1 = self.up4(x2, x1)
        x0 = self.seg_up(x1, x)
        # Detection heads.
        outs = []
        out = {}
        y = self.seg(x0)
        out['seg'] = y
        outs.append(out)
        # Decoding boxes in gpu when doing infer.
        if self.phase != 'train':
            seg_output = F.softmax(y, dim=1)
            seg_pred = seg_output.argmax(dim=1)
            val_out = {'seg': seg_pred, 'seg_prob': seg_output}
            return [val_out]
        return outs

def get_unet(seg_class, phase='train'):
    model = UNet(seg_class, phase)
    return model
    