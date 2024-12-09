import torch
import torch.nn as nn
from utils.general import check_version

__all__ = ['FRMHead']

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))



class Proto(nn.Module):
    # YOLOv5 mask Proto module for segmentation models
    def __init__(self, c1, c_=256, c2=32):  # ch_in, number of protos, number of masks
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))



class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)

class PCRC(nn.Module):
    def __init__(self):
        super().__init__()
        self.C1 = nn.Conv2d(3, 512 * 3, kernel_size=1)
        self.R1 = nn.Upsample(None, 2, 'nearest')  # 上采样扩充2倍采用邻近扩充
        self.mcrc = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512 * 3, 512 * 3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 * 3, 512 * 3, kernel_size=3, padding=1),
        )
        self.acrc = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512 * 3, 512 * 3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 * 3, 512 * 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x1 = self.C1(x)
        x2 = self.mcrc(x1)
        x3 = self.acrc(x1)
        return self.R1(x2) + self.R1(x3)

class FRM(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        super().__init__()
        channel1 = ch[0]  # 64
        channel2 = ch[1]  # 128
        channel3 = ch[2]  # 256
        self.split_stride = channel2
        self.R1 = nn.Upsample(None, 2, 'nearest')  #上采样扩充2倍采用邻近扩充
        self.R3 = nn.MaxPool2d(kernel_size=2, stride=2)   #下采样使用最大池化
        self.C1 = nn.Conv2d(channel1 + channel2 + channel3, 3, kernel_size=1)

        self.C2 = nn.Conv2d(channel2, channel3, kernel_size=1, stride=1)
        self.C3 = nn.Conv2d(channel2, channel1, kernel_size=1, stride=1)
        self.C4 = nn.Conv2d(1, channel1, kernel_size=1, stride=1)
        self.C5 = nn.Conv2d(1, channel2, kernel_size=1, stride=1)
        self.C6 = nn.Conv2d(1, channel3, kernel_size=1, stride=1)
        self.pcrc = PCRC()

    def forward(self, x):

        x0 = self.R1(x[0])
        x2 = self.R3(x[2])
        input = torch.cat((x0, x[1], x2), 1)
        x1 = self.C1(input)
        Conv_1_1 = torch.split(torch.softmax(x1, dim=0), 1, 1)
        Conv_1_2 = torch.split(self.pcrc(x1), self.split_stride, 1)
        input1 = (self.C2(Conv_1_2[0]) * x0)
        input2 = (x0 * self.C6(Conv_1_1[0]))
        y0 = input1 + input2
        y1 = (x[1] * self.C5(Conv_1_1[1])) + (Conv_1_2[1] * x[1])
        y2 = (x2 * self.C4(Conv_1_1[2])) + (self.C3(Conv_1_2[2]) * x2)

        y0 = self.R3(y0)
        y2 = self.R1(y2)

        return [y2, y1, y0]


class FRMHead(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True, multiplier=0.25, rfb=False):  # detection layer
        super().__init__()

        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch * 3)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.FRM = FRM(ch=ch)

    def forward(self, x):
        x.reverse()
        x = self.FRM(x)
        z = []  # inference output
        for i in range(self.nl):

            x[i] = self.m[i](x[i])  # conv

            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, FRMHead):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid



class Segment_FRM(FRMHead):
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = FRMHead.forward

    def forward(self, x):
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])



