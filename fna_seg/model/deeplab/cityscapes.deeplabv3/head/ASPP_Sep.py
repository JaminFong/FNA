import torch.nn as nn
import torch
import torch.nn.functional as F


class ASPP_Sep(nn.Module):
    
    def __init__(self, dim_in, dim_out, BatchNorm=None, rate=1):
        super(ASPP_Sep, self).__init__()
        self.branch1 = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=False),
                BatchNorm(dim_out),
                nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
                SepConv(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=False, BatchNorm=BatchNorm,),
        )

        self.branch3 = nn.Sequential(
                SepConv(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=False, BatchNorm=BatchNorm,),
        )

        self.branch4 = nn.Sequential(
                SepConv(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=False, BatchNorm=BatchNorm,),
        )

        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)
        self.branch5_bn = BatchNorm(dim_out)
        self.branch5_relu = nn.ReLU(inplace=True)
        self.conv_cat = nn.Sequential(
                nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0, bias=False),
                BatchNorm(dim_out),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1)
        )

    def forward(self, x):
        [b,c,row,col] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = torch.mean(x,2,True)
        global_feature = torch.mean(global_feature,3,True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row,col), None, 'bilinear', True)
        
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result

class SepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False, BatchNorm=nn.BatchNorm2d,
                activate_first=False, inplace=True):
        super(SepConv, self).__init__()
        self.relu0 = nn.ReLU(inplace=True)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                                   bias=bias)
        self.bn1 = BatchNorm(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        self.bn2 = BatchNorm(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.activate_first = activate_first

    def forward(self, x):
        if self.activate_first:
            x = self.relu0(x)
        x = self.depthwise(x)
        x = self.bn1(x)
        if not self.activate_first:
            x = self.relu1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        if not self.activate_first:
            x = self.relu2(x)
        return x
