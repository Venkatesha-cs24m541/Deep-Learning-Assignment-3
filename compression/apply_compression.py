import torch.nn as nn
from .quant_layers import QuantConv2d, QuantLinear


def replace_modules(model, w_bits, a_bits):
    """
    Recursively replace Conv2d and Linear layers
    """
    for name, module in model.named_children():

        if isinstance(module, nn.Conv2d):
            quant_conv = QuantConv2d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=(module.bias is not None),
                w_bits=w_bits,
                a_bits=a_bits
            )
            quant_conv.weight.data = module.weight.data.clone()
            if module.bias is not None:
                quant_conv.bias.data = module.bias.data.clone()

            setattr(model, name, quant_conv)

        elif isinstance(module, nn.Linear):
            quant_fc = QuantLinear(
                module.in_features,
                module.out_features,
                bias=(module.bias is not None),
                w_bits=w_bits,
                a_bits=a_bits
            )
            quant_fc.weight.data = module.weight.data.clone()
            if module.bias is not None:
                quant_fc.bias.data = module.bias.data.clone()

            setattr(model, name, quant_fc)

        else:
            replace_modules(module, w_bits, a_bits)
