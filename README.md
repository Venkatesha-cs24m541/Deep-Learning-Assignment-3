# Deep-Learning-Assignment-3
---
### CS6886W — System Engineering for Deep Learning
### Assignment 3
### Department of Computer Science and Engineering, IIT Madras

---
## Overview
This assignment focuses on training MobileNet-v2 on CIFAR-10 and then applying model compression techniques to reduce model size while retaining accuracy. <br>
Both accuracy and compression effectiveness will be evaluated. <br>

---
## Environment Setup
 Clone the repository
git clone https://github.com/Venkatesha-cs24m541/Deep-Learning-Assignment-3

### Running from Drive
1. Upload all the files to a folder in Drive
2. %cd /content/drive/MyDrive/Semester_3/Deep_Learning/Assignment_3/
3. !pip install torch torchvision wandb matplotlib numpy wandb
4. import wandb
5. wandb.login()
6. To run the baseline test - !python train_baseline.py
7. To run the baseline test - !python train_pretrained.py
8. To run the sweep - !wandb sweep /sweeps/sweep.yaml

### Running Directly
1. pip install torch torchvision wandb matplotlib numpy wandb
2. python
3. import wandb
4. wandb.login()
5. python train_baseline.py
6. python train_pretrained.py
7. wandb sweep /sweeps/sweep.yaml

---
 ## File Structure
 vgg6-cifar10-experiments <br>
 ┣ compression <br> 
   ┣ apply_compression.py #Apply the compression <br>
   ┣ quant_layers.py #Qualntisation layers <br>
   ┣ quant_utils.py #Quantisation Utilities <br>
<br>
 ┣ sweeps <br>
   ┣ sweep.yaml # W&B sweep configuration <br>
<br>
 ┣ utils # Utilities needed <br>
   ┣ activation_size.py #Activation sizes <br>
   ┣ cifar.py #Transforms used for CIFAR <br>
   ┣ memory.py #Memory of model and f32 size <br>
<br>
 ┣ README.md # This documentation <br>
<br> 
 ┣ results/ # The report plots <br>
   ┣ Results.csv #Results of quantisation <br>
   ┣ Baseline_Accuracy.png #Baseline Accuracy PNG <br>
   ┣ Scatter_Plot.png #Quantisation scatter plot <br>
   ┣ Scatter_Plot_1.png #Quantisation scatter plot 1 <br>
<br>   
 ┣ train_baseline.py/ # Train the baseline from scratch <br>
 ┣ train_pretrained.py/ # Train the baseline using pretrained model <br>
 ┗ train_quantized.py/ # Train the quantised model <br>

---
## Transforms used
Transforms Used :-
def get_cifar10(batch_size=128, num_workers=2):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])

---
## Baseline (Based on W&B Parallel Plot)
From scratch - baseline_scratch_2.pth - epoch:300, test_acc:0.9019 <br>
With ImageNet pretraining - baseline_pretrained_1.pth - epoch:150, test_acc:0.9344 <br>

---
## Compression best result
activation_quant_bits - 8 <br>
weight_quant_bits - 8 <br>
model_compression_ratio - 3.998870073668690 <br>
weight_compression_ratio - 4 <br>
activation_compression_ratio - 4 <br>
quant_act_MB - 20857.21969604490 <br>
quant_weight_MB - 2.133668899536130 <br>
accuracy - 0.9329 <br>

---
## Reproducibility Details
Random Seed: 42 <br>
Hardware: NVIDIA GPU (Colab / RTX 3060 recommended) <br>
Framework: PyTorch 2.2+ <br>
Dataset: CIFAR-10 (auto-downloaded via torchvision) <br>
Logging: Weights & Biases (wandb) <br>

All runs are reproducible using the same random seed and environment configuration.

