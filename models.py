"""
Neural Network Models for Adversarial Training.

This module contains the PreActResNet18 architecture, which is the standard
architecture used in catastrophic overfitting research.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""
    
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes,
                    kernel_size=1, stride=stride, bias=False
                )
            )
        else:
            self.shortcut = nn.Sequential()
    
    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        out += shortcut
        return out


class PreActResNet18(nn.Module):
    """
    PreActResNet18 for CIFAR-10.
    
    This is the standard architecture used in catastrophic overfitting research.
    It uses pre-activation (BatchNorm + ReLU before convolution) which has been
    shown to be more stable for adversarial training.
    
    Args:
        num_classes (int): Number of output classes. Default: 10 (CIFAR-10)
    
    Input:
        x (torch.Tensor): Input images of shape (batch_size, 3, 32, 32)
    
    Output:
        torch.Tensor: Logits of shape (batch_size, num_classes)
    """
    
    def __init__(self, num_classes=10):
        super(PreActResNet18, self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(PreActBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(PreActBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(PreActBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(PreActBlock, 512, 2, stride=2)
        self.bn = nn.BatchNorm2d(512 * PreActBlock.expansion)
        self.linear = nn.Linear(512 * PreActBlock.expansion, num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def get_model(architecture='PreActResNet18', num_classes=10, device='cpu'):
    """
    Factory function to get a model by name.
    
    Args:
        architecture (str): Model architecture name
        num_classes (int): Number of output classes
        device (str or torch.device): Device to place the model on
    
    Returns:
        nn.Module: The requested model
    """
    if architecture == 'PreActResNet18':
        model = PreActResNet18(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return model.to(device)
