import torch
from .dropped_model import Dropped_Network

class SSDLite_Dropped_Network(Dropped_Network):
    def __init__(self, super_model):
        super(SSDLite_Dropped_Network, self).__init__(super_model)
        self.extra = super_model.extra 


    def forward(self, inputs):
        net_sub_obj = torch.tensor(0., dtype=torch.float).cuda()

        outs = []
        results = self.blocks[0](inputs)
        results = self.blocks[1](results)
        for i, block in enumerate(self.blocks[2:]):
            if i+2 == self.output_indices[0]:
                results, block_sub_obj, head_data = block(results, self.alphas_normal[i], 
                                        self.alphas_reduce[i], self.alpha_index[i], 
                                        self.sub_obj_list[i])
                outs.append(head_data)
            elif i+2==len(self.blocks)-1:
                results = block(results)
            else:
                results, block_sub_obj = block(results, self.alphas_normal[i], 
                                        self.alphas_reduce[i], self.alpha_index[i], 
                                        self.sub_obj_list[i])

            net_sub_obj += block_sub_obj
            if i+2 in self.output_indices[1:]:
                outs.append(results)

        for i, layer in enumerate(self.extra):
            results = layer(results)
            if i % 2 == 1:
                outs.append(results)

        if len(outs) == 1:
            return outs[0], net_sub_obj
        else:
            return tuple(outs), net_sub_obj
