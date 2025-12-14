import torch

class ActivationSizeTracker:
    def __init__(self, bits):
        self.bits = bits
        self.total_bits = 0
        self.handles = []

    def hook(self, module, inp, out):
        if isinstance(out, torch.Tensor):
            self.total_bits += out.numel() * self.bits

    def attach(self, model):
        for m in model.modules():
            self.handles.append(m.register_forward_hook(self.hook))

    def detach(self):
        for h in self.handles:
            h.remove()

    def get_size_bytes(self):
        return self.total_bits / 8
