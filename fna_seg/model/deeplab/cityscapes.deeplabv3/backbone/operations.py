import torch
import torch.nn as nn
import torch.nn.functional as F

OPS = {
    'k3_e3': lambda C_in, C_out, stride, dilation, affine, track: MBBlock(C_in, C_out, 3, stride, 3, 1, dilation, affine=affine, track = track),
    'k3_e6': lambda C_in, C_out, stride, dilation, affine, track: MBBlock(C_in, C_out, 3, stride, 6, 1, dilation, affine=affine, track = track),
    'k5_e1': lambda C_in, C_out, stride, dilation, affine, track: MBBlock(C_in, C_out, 5, stride, 1, 1, dilation, affine=affine, track = track),
    'k3_e1': lambda C_in, C_out, stride, dilation, affine, track: MBBlock(C_in, C_out, 3, stride, 1, 1, dilation, affine=affine, track = track),
    'k5_e3': lambda C_in, C_out, stride, dilation, affine, track: MBBlock(C_in, C_out, 5, stride, 3, 1, dilation, affine=affine, track = track),
    'k5_e6': lambda C_in, C_out, stride, dilation, affine, track: MBBlock(C_in, C_out, 5, stride, 6, 1, dilation, affine=affine, track = track),
    'k7_e1': lambda C_in, C_out, stride, dilation, affine, track: MBBlock(C_in, C_out, 7, stride, 1, 1, dilation, affine=affine, track = track),
    'k7_e3': lambda C_in, C_out, stride, dilation, affine, track: MBBlock(C_in, C_out, 7, stride, 3, 1, dilation, affine=affine, track = track),
    'k7_e6': lambda C_in, C_out, stride, dilation, affine, track: MBBlock(C_in, C_out, 7, stride, 6, 1, dilation, affine=affine, track = track),
    'skip': lambda C_in, C_out, stride, dilation, affine, track: Skip(C_in, C_out, stride, affine=affine),
}


def comp_padding(kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad = pad_total // 2
    return pad

class MBBlock(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride,
              expansion, group, dilation, affine = False, track = False, bias=False):
        super(MBBlock, self).__init__()
        bias_flag = False
        self.dilation = dilation
        self.kernel_size = kernel_size
        pad = comp_padding(kernel_size, dilation)
        if expansion != 1:
            self.op = nn.Sequential(
                nn.Conv2d(C_in, C_in * expansion, 1, stride=1, padding=0,
                          groups=group, bias=bias_flag),
                nn.BatchNorm2d(C_in * expansion,affine=affine,track_running_stats=track),
                nn.ReLU6(inplace=True),
                nn.Conv2d(C_in * expansion, C_in * expansion, kernel_size, stride=stride,
                          padding=pad, dilation=dilation, groups=C_in * expansion, bias=bias_flag),
                nn.BatchNorm2d(C_in * expansion,affine=affine,track_running_stats=track),
                nn.ReLU6(inplace=True),
                nn.Conv2d(C_in * expansion, C_out, 1, stride=1, padding=0,
                          groups=group, bias=bias_flag),
                nn.BatchNorm2d(C_out,affine=affine,track_running_stats=track),
            )
        else:
            self.op = nn.Sequential(
                nn.Conv2d(C_in * expansion, C_in * expansion, kernel_size, stride=stride,
                          padding=pad, dilation=dilation, groups=C_in * expansion, bias=bias_flag),
                nn.BatchNorm2d(C_in * expansion,affine=affine,track_running_stats=track),
                nn.ReLU6(inplace=True),
                nn.Conv2d(C_in * expansion, C_out, 1, stride=1, padding=0,
                          groups=group, bias=bias_flag),
                nn.BatchNorm2d(C_out,affine=affine,track_running_stats=track),
            )
        res_flag = ((C_in == C_out) and (stride == 1))
        self.res_flag = res_flag

    def forward(self, x):
        if self.res_flag:
            return self.op(x) + x
        else:
            return self.op(x)  # + self.trans(x)


def Skip(C_in, C_out, stride, affine):
    if C_in == C_out and stride == 1:
        return Identity()
    elif C_in != C_out and stride == 1:
        return Conv1_1(C_in, C_out, affine)
    elif C_in != C_out and stride == 2:
        return FactorizedReduce(C_in, C_out, affine)
    else:
        raise ValueError("operations.py,line 29")


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Conv1_1(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super(Conv1_1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(C_in, C_out, 1, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


def fixed_padding(inputs):
    padded_inputs = F.pad(inputs, (0, 1, 0, 1))
    return padded_inputs

class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0

        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if x.size()[3] % 2 != 0:
            x = fixed_padding(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        return out


def conv_dw(inp, oup, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size=3, stride=stride, padding=padding, groups=inp, bias=False),
        nn.BatchNorm2d(inp, eps=1e-03),
        nn.ReLU6(inplace=True),

        nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(oup, eps=1e-03),
    )


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )