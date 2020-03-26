import torch.nn as nn

from mmcv.cnn import constant_init, normal_init
from mmdet.models.registry import BACKBONES

from .derive_blocks import derive_blocks
from .operations import Conv1_1, conv_dw
from tools.apis.param_remap import remap_for_paramadapt


@BACKBONES.register_module
class FNA_SSDLite(nn.Module):
    extra_setting = {
        300: (256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 64, 'S', 128),
        512: (256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128),
    }

    def __init__(self, input_size, net_config, out_feature_indices=(6, 8)):
        super(FNA_SSDLite, self).__init__()
        assert input_size in (300, 512)
        self.input_size = input_size
        self.out_feature_indices = out_feature_indices
        self.inplanes = 1280

        self.blocks, parsed_net_config = derive_blocks(net_config)
        self.blocks.append(Conv1_1(parsed_net_config[-1][0][1], self.inplanes))
        self.extra = self._make_extra_layers(self.extra_setting[input_size])

    def forward(self, x):
        outs = []
        for i, block in enumerate(self.blocks):
            if i==self.out_feature_indices[0]:
                for j, layer in enumerate(block.layers[0].op):
                    x = layer(x)
                    if j==2:
                        outs.append(x)
                for layer in block.layers[1:]:
                    x = layer(x)
            else:
                x = block(x)
                if i in self.out_feature_indices:
                    outs.append(x)

        for i, layer in enumerate(self.extra):
            x = layer(x)
            if i % 2 == 1:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(FNA_SSDLite, self).train(mode)

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.03)
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if pretrained is not None and pretrained.use_load:
            model_dict = remap_for_paramadapt(pretrained.load_path, self.state_dict(), 
                                                pretrained.seed_num_layers)
            self.load_state_dict(model_dict)
        
        for m in self.extra.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.03)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    def _make_extra_layers(self, outplanes):
        layers = []
        kernel_sizes = (1, 3)
        num_layers = 0
        outplane = None
        for i in range(len(outplanes)):
            if self.inplanes == 'S':
                self.inplanes = outplane
                continue
            k = kernel_sizes[num_layers % 2]
            if outplanes[i] == 'S':
                outplane = outplanes[i + 1]
                conv = conv_dw(
                    self.inplanes, outplane, stride=2, padding=1)
            else:
                outplane = outplanes[i]
                conv = nn.Sequential(
                    nn.Conv2d(self.inplanes, outplane, k, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(outplane),
                    nn.ReLU6(inplace=True)
                )
            layers.append(conv)
            self.inplanes = outplanes[i]
            num_layers += 1

        return nn.Sequential(*layers)
