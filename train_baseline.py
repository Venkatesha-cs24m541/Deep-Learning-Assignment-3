import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import wandb

from utils.cifar import get_cifar10


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
    wandb.init(project="cs6886-assignment3", name="baseline_scratch")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = get_cifar10()

    # MobileNetV2 from torchvision (NO pretraining)
    model = models.mobilenet_v2(weights=None)

    # CIFAR-10 FIX
    model.features[0][0].stride = (1, 1)

    print(model.features[0][0])

    # Replace classifier for CIFAR-10
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, 10
    )

    model.to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=300
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(300):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        acc = evaluate(model, test_loader, device)
        scheduler.step()

        wandb.log({
            "epoch": epoch,
            "test_acc": acc
        })

    torch.save(model.state_dict(), "baseline_scratch.pth")


if __name__ == "__main__":
    main()
