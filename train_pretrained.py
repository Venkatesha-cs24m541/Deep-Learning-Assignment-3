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
    wandb.init(project="cs6886-assignment3", name="baseline_pretrained_1")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = get_cifar10()

    # ImageNet pretrained MobileNetV2
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")

    # CIFAR-10 FIX
    model.features[0][0].stride = (1, 1)

    print(model.features[0][0])

    # Replace classifier
    model.classifier[1] = nn.Linear(
      model.classifier[1].in_features, 10
    )

    # Replace classifier
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, 10
    )

    model.to(device)

    # Lower LR for finetuning
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=150
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(150):
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

    torch.save(model.state_dict(), "baseline_pretrained_1.pth")


if __name__ == "__main__":
    main()
