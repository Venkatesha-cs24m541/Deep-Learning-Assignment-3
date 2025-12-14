import torch

def symmetric_quantize(x, bits):
    """
    Symmetric uniform quantization.
    Returns quantized tensor (int) and scale.
    """
    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1

    max_val = x.abs().max()
    scale = max_val / qmax if max_val > 0 else torch.tensor(1.0, device=x.device)

    x_int = torch.round(x / scale)
    x_int = torch.clamp(x_int, qmin, qmax)

    return x_int, scale


def symmetric_dequantize(x_int, scale):
    return x_int * scale
