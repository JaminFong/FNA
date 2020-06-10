import torch.nn as nn
import math
from .derive_blocks import derive_blocks
from .operations import Conv1_1


class ImageNetModel(nn.Module):
    def __init__(self, net_config):
        super(ImageNetModel, self).__init__()
        self.blocks, parsed_net_config = derive_blocks(net_config)
        self.blocks.append(Conv1_1(parsed_net_config[-1][0][1], 1280))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, 1000)
        self.init_model()
        self.set_bn_param(0.1, 1e-3)

    def forward(self, x, stat=None):
        for block in self.blocks:
            x = block(x)

        out = self.global_pooling(x)
        logits = self.classifier(out.view(out.size(0),-1))
        return logits

    def init_model(self, model_init='he_fout', init_div_groups=True):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if model_init == 'he_fout':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif model_init == 'he_fin':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                else:
                    raise NotImplementedError
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def set_bn_param(self, bn_momentum, bn_eps):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = bn_momentum
                m.eps = bn_eps
        return

