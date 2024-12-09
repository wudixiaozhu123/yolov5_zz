import torch
import torch.nn as nn
from utils.general import check_version

__all__ = ['CLLAHead', 'Segment_CLLA']

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

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
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

class CLLA(nn.Module):
    def __init__(self, range, c):
        super().__init__()
        self.c_ = c
        self.q = nn.Linear(self.c_, self.c_)
        self.k = nn.Linear(self.c_, self.c_)
        self.v = nn.Linear(self.c_, self.c_)
        self.range = range
        self.attend = nn.Softmax(dim = -1)

    def forward(self, x1, x2):
        b1, c1, w1, h1 = x1.shape
        b2, c2, w2, h2 = x2.shape
        assert b1 == b2 and c1 == c2

        x2_ = x2.permute(0, 2, 3, 1).contiguous().unsqueeze(3)
        pad = int(self.range / 2 - 1)
        padding = nn.ZeroPad2d(padding=(pad, pad, pad, pad))
        x1 = padding(x1)

        local = []
        for i in range(int(self.range)):
            for j in range(int(self.range)):
                tem = x1
                tem = tem[..., i::2, j::2][..., :w2, :h2].contiguous().unsqueeze(2)
                local.append(tem)
        local = torch.cat(local, 2)

        x1 = local.permute(0, 3, 4, 2, 1)

        q = self.q(x2_)
        k, v = self.k(x1), self.v(x1)

        dots = torch.sum(q * k / self.range, 4)
        irr = torch.mean(dots, 3).unsqueeze(3) * 2 - dots
        att = self.attend(irr)

        out = v * att.unsqueeze(4)
        out = torch.sum(out, 3)
        out = out.squeeze(3).permute(0, 3, 1, 2).contiguous()
        # x2 = x2.squeeze(3).permute(0, 3, 1, 2).contiguous()
        return (out + x2) / 2
        # return out

class CLLABlock(nn.Module):
    def __init__(self, range=2, ch=256, ch1=128, ch2=256, out=0):
        super().__init__()
        self.range = range
        self.c_ = ch
        self.cout = out
        self.conv1 = nn.Conv2d(ch1, self.c_, 1)
        self.conv2 = nn.Conv2d(ch2, self.c_, 1)

        self.att = CLLA(range = range, c = self.c_)

        self.det = nn.Conv2d(self.c_, out, 1)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)

        f = self.att(x1, x2)

        return self.det(f)


class CLLAHead(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()

        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.det = CLLABlock(range=2, ch=ch[1], ch1=ch[0], ch2=ch[1], out=self.no * 3)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch * 3)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)


    def forward(self, x):
        y1 = x[0]
        z = []  # inference output
        for i in range(self.nl):
            if i == 1:
                x[i] = self.det(y1, x[1])
            else:
                x[i] = self.m[i](x[i])  # conv

            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment_CLLA):  # (boxes + masks)
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

class Segment_CLLA(CLLAHead):
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = CLLAHead.forward

    def forward(self, x):
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])



if __name__ == "__main__":

    # Generating Sample image
    image1 = (16, 64, 80, 80)
    image2 = (16, 128, 40, 40)
    image3 = (16, 256, 20, 20)

    image1 = torch.rand(image1)
    image2 = torch.rand(image2)
    image3 = torch.rand(image3)
    image = [image1, image2, image3]
    channel = (64, 128, 256)
    num_classes = 80
    num_layers = 3
    use_dfl = True
    reg_max = 16

    head = CLLAHead(num_classes, channel)

    out = head(image)
    print(len(out))

