import logging
from collections import Iterable


class ArchGenerate_FNA(object):
    def __init__(self, super_network):
        self.primitives_normal = super_network.primitives_normal
        self.primitives_reduce = super_network.primitives_reduce
        if hasattr(super_network, 'primitives_for_head'):
            self.primitives_for_head = super_network.primitives_for_head
        self.num_layers = super_network.num_layers
        self.chs = super_network.search_params.net_scale.chs

    def update_arch_params(self, alphas):
        self.alphas = alphas
    
    def derive_archs(self, alphas, logger=None):
        flat = lambda t: [x for sub in t for x in flat(sub)] if isinstance(t, Iterable) else [t]
        self.update_arch_params(alphas)
        def _parse(weights):
            assert len(alphas) == sum(self.num_layers)

            final_stages = []
            final_stages.append(['k3_e1'])
            count = 0
            for num_layer in self.num_layers:
                stage = []
                for _ in range(num_layer):
                    weight = weights[count]
                    if len(weight) == len(self.primitives_reduce):
                        op = self.primitives_reduce[weight.index(max(weight))]
                    elif len(weight) == len(self.primitives_normal):
                        op = self.primitives_normal[weight.index(max(weight))]
                    elif hasattr(self, 'primitives_for_head'):
                        op = self.primitives_for_head[weight.index(max(weight))]
                    else:
                        raise ValueError
                    stage.append(op)
                    count += 1
                final_stages.append(stage)

            final_code = []
            for i, stage in enumerate(final_stages):
                if i in [1,2,3,5]:
                    stride = 2
                else:
                    stride = 1
                final_code.append([[self.chs[i], self.chs[i+1]], stage, stride])
            return ('|\n'.join(map(str, final_code)))

        net_config = _parse(alphas)
        logging.debug(net_config)
        return net_config