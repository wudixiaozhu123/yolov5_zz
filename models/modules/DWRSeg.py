import torch
import torch.nn as nn

__all__ = ['C3_DWRSeg']

class Conv(nn.Module):
    # 包含BN和ReLU
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DWR(nn.Module):
    def __init__(self, c) -> None:
        super().__init__()

        self.conv_3x3 = Conv(c, c, 3, padding=1)

        self.conv_3x3_d1 = Conv(c, c, 3, padding=1, dilation=1)
        self.conv_3x3_d3 = Conv(c, c, 3, padding=3, dilation=3)
        self.conv_3x3_d5 = Conv(c, c, 3, padding=5, dilation=5)

        self.conv_1x1 = Conv(c * 3, c, 1)

    def forward(self, x):
        x_ = self.conv_3x3(x)
        x1 = self.conv_3x3_d1(x_)
        x2 = self.conv_3x3_d3(x_)
        x3 = self.conv_3x3_d5(x_)

        x_out = torch.cat([x1, x2, x3], dim=1)
        x_out = self.conv_1x1(x_out) + x
        return x_out


class DWRSeg_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=1, dilation=1):
        super().__init__()
        self.conv = Conv(in_channels, out_channels, 1)

        self.dcnv3 = DWR(out_channels)

        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)

        x = self.dcnv3(x)

        x = self.gelu(self.bn(x))
        return x


class Bottleneck_DWRSeg(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = DWRSeg_Conv(c_, c2, k[1], 1, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3_DWRSeg(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck_DWRSeg(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))