import torch
import numpy as np

def model_weight_size(model, w_bits):
    total_bits = 0
    scale_bits = 0

    for p in model.parameters():
        total_bits += p.numel() * w_bits
        scale_bits += 32  # one float scale per tensor

    return (total_bits + scale_bits) / 8


def fp32_model_size(model):
    total_bits = 0
    for p in model.parameters():
        total_bits += p.numel() * 32
    return total_bits / 8
