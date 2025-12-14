import torch
import torch.nn as nn
import torch.nn.functional as F
from .quant_utils import symmetric_quantize, symmetric_dequantize


class QuantConv2d(nn.Conv2d):
    def __init__(self, *args, w_bits=8, a_bits=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.w_bits = w_bits
        self.a_bits = a_bits

    def forward(self, x):
        x_q, x_scale = symmetric_quantize(x, self.a_bits)
        w_q, w_scale = symmetric_quantize(self.weight, self.w_bits)

        x_dq = symmetric_dequantize(x_q, x_scale)
        w_dq = symmetric_dequantize(w_q, w_scale)

        return F.conv2d(
            x_dq, w_dq, self.bias,
            self.stride, self.padding,
            self.dilation, self.groups
        )


class QuantLinear(nn.Linear):
    def __init__(self, *args, w_bits=8, a_bits=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.w_bits = w_bits
        self.a_bits = a_bits

    def forward(self, x):
        x_q, x_scale = symmetric_quantize(x, self.a_bits)
        w_q, w_scale = symmetric_quantize(self.weight, self.w_bits)

        x_dq = symmetric_dequantize(x_q, x_scale)
        w_dq = symmetric_dequantize(w_q, w_scale)

        return F.linear(x_dq, w_dq, self.bias)
