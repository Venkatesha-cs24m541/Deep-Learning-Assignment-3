import torch
import torch.nn as nn
import torchvision.models as models
import wandb

from utils.cifar import get_cifar10
from compression.apply_compression import apply_compression
from utils.activation_size import ActivationTracker
from utils.memory import fp32_size, quant_weight_size


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def main():
    wandb.init(project="cs6886-assignment3")

    # 🔹 Read sweep parameters
    w_bits = int(wandb.config.weight_quant_bits)
    a_bits = int(wandb.config.activation_quant_bits)
    model_init = wandb.config.model_init  # "scratch" or "pretrained"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test_loader = get_cifar10()

    # 🔹 Build model
    model = models.mobilenet_v2(weights=None)

    # CIFAR-10 fix (VERY IMPORTANT)
    model.features[0][0].stride = (1, 1)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, 10
    )

    # 🔹 Load correct checkpoint
    if model_init == "pretrained":
        ckpt_path = "baseline_pretrained.pth"
    else:
        ckpt_path = "baseline_scratch.pth"

    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

    # 🔹 Apply custom quantization
    apply_compression(model, w_bits, a_bits)
    model.to(device)

    # 🔹 Activation memory tracking
    tracker = ActivationTracker(a_bits)
    tracker.attach(model)

    acc = evaluate(model, test_loader, device)

    tracker.detach()

    # 🔹 Memory calculations
    fp32_mb = fp32_size(model) / (1024 ** 2)
    quant_weight_mb = quant_weight_size(model, w_bits) / (1024 ** 2)
    activation_mb = tracker.total_bits / 8 / (1024 ** 2)

    # 🔹 Compression ratios (derived)
    weight_cr = 32 / w_bits
    model_cr = fp32_mb / quant_weight_mb

    # 🔹 Log everything
    wandb.log({
        "model_init": model_init,
        "weight_bits": w_bits,
        "activation_bits": a_bits,
        "accuracy": acc,

        # sizes
        "fp32_model_MB": fp32_mb,
        "quant_weight_MB": quant_weight_mb,
        "activation_MB": activation_mb,

        # compression ratios
        "weight_compression_ratio": weight_cr,
        "model_compression_ratio": model_cr,
    })


if __name__ == "__main__":
    main()
