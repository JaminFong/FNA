import logging
from collections import Iterable


class ArchGenerate_FNA(object):
    def __init__(self, super_network):
        self.primitives_normal = super_network.primitives_normal
        self.primitives_reduce = super_network.primitives_reduce
        if hasattr(super_network, 'primitives_special'):
            self.primitives_special = super_network.primitives_special

    def update_arch_params(self, alphas):
        self.alphas = alphas
    
    def derive_archs(self, alphas, logger=None):
        flat = lambda t: [x for sub in t for x in flat(sub)] if isinstance(t, Iterable) else [t]
        self.update_arch_params(alphas)
        def _parse(weights):
            # gene = ['k3_e1_g1']
            # n = [4, 4, 4, 4, 4, 1]
            gene = []
            n = [3,5,7,4]

            i = 0
            for _n in n:
                for idx in range(_n):
                    W = flat(weights[i])
                    k_best = None
                    for k in range(len(W)):
                        if k_best is None or W[k] > W[k_best]:
                            k_best = k
                    i += 1
                    if idx == 0 and len(W) == 3:
                        gene.append(self.primitives_special[k_best])
                    elif idx == 0 and len(W) != 3:
                        gene.append(self.primitives_reduce[k_best])
                    else:
                        gene.append(self.primitives_normal[k_best])

            return gene

        gene_normal = _parse(alphas)
        print(gene_normal)
        return gene_normal