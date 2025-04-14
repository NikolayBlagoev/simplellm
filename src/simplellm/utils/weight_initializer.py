import torch.nn as nn
def initialize_weights(module):
    std = 2e-2
    
    if isinstance(module, nn.Linear):
        print(module)
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        print(module)
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()

class IterableModule(object):
    def _initialize(self, initializer_method = initialize_weights):
        for _,module in self.named_modules():
            
            initializer_method(module)
            # if hasattr(module, "_initialize"):
            #     module._initialize(initializer_method)
        initializer_method(self)


