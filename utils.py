"""
Utility functions for adversarial training experiments.

This module contains helper functions for:
- Data loading and preprocessing
- Model evaluation
- Participation Ratio calculation
- Training utilities
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np

from src.attacks import fgsm_attack, pgd_attack


def load_cifar10(batch_size=128, num_workers=2, data_dir='./data'):
    """
    Load CIFAR-10 dataset with standard preprocessing.
    
    Args:
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker processes for data loading
        data_dir (str): Directory to store/load the dataset
    
    Returns:
        tuple: (trainloader, testloader)
    """
    # Training transformations (with data augmentation)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    # Test transformations (no augmentation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Load datasets
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return trainloader, testloader


def compute_participation_ratio(grad):
    """
    Compute Participation Ratio (PR1) = (||grad||_1 / ||grad||_2)^2.
    
    This metric measures gradient concentration:
    - Lower PR: gradients concentrated in few dimensions
    - Higher PR: gradients spread across many dimensions
    - Sharp PR drops correlate with CO onset
    
    Args:
        grad (torch.Tensor): Gradient tensor of shape (batch_size, ...)
    
    Returns:
        float: Mean participation ratio across the batch
    
    References:
        Mehouachi et al. (2025) "A Noiseless lp Norm Solution for
        Fast Adversarial Training"
    """
    grad_flat = grad.view(grad.size(0), -1)
    l1 = grad_flat.abs().sum(dim=1)
    l2 = grad_flat.norm(2, dim=1).clamp(min=1e-9)
    pr = (l1 / l2) ** 2
    return pr.mean().item()


def evaluate_robustness(model, testloader, epsilon, attack_type='fgsm', num_steps=50):
    """
    Evaluate model robustness against a specified attack.
    
    Args:
        model (nn.Module): The model to evaluate
        testloader (DataLoader): Test data loader
        epsilon (float): Perturbation budget
        attack_type (str): Type of attack ('clean', 'fgsm', or 'pgd')
        num_steps (int): Number of PGD steps (only used if attack_type='pgd')
    
    Returns:
        float: Accuracy percentage
    """
    model.eval()
    correct = 0
    total = 0
    device = next(model.parameters()).device
    
    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Generate adversarial examples
        if attack_type == 'fgsm':
            X_adv = fgsm_attack(model, inputs, targets, epsilon)
        elif attack_type == 'pgd':
            alpha = epsilon / 4  # Standard step size
            X_adv = pgd_attack(
                model, inputs, targets, epsilon, alpha, num_steps, restarts=1
            )
        else:  # clean
            X_adv = inputs
        
        # Evaluate accuracy
        with torch.no_grad():
            outputs = model(X_adv)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


def compute_gradient_norm(model):
    """
    Compute the L2 norm of the model's gradients.
    
    Args:
        model (nn.Module): The model with computed gradients
    
    Returns:
        float: Total gradient norm
    """
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def detect_catastrophic_overfitting(
    robustness_gap,
    test_pgd_acc,
    fgsm_train_acc,
    test_pgd_acc_threshold=5,
    gap_threshold=90,
    train_test_gap_threshold=60
):
    """
    Detect if catastrophic overfitting has occurred.
    
    Args:
        robustness_gap (float): FGSM test - PGD test accuracy gap
        test_pgd_acc (float): PGD-50 test accuracy
        fgsm_train_acc (float): FGSM training accuracy
        test_pgd_acc_threshold (float): Threshold for PGD accuracy
        gap_threshold (float): Threshold for robustness gap
        train_test_gap_threshold (float): Threshold for train-test gap
    
    Returns:
        bool: True if CO is detected
    """
    co_detected = (
        robustness_gap > gap_threshold and
        test_pgd_acc < test_pgd_acc_threshold
    )
    return co_detected


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(
    epoch,
    model,
    optimizer,
    scheduler,
    metrics,
    filename
):
    """
    Save a model checkpoint.
    
    Args:
        epoch (int): Current epoch
        model (nn.Module): The model
        optimizer (Optimizer): The optimizer
        scheduler (LRScheduler): The learning rate scheduler
        metrics (dict): Dictionary of metrics to save
        filename (str): Checkpoint filename
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }, filename)


def load_checkpoint(filename, model, optimizer=None, scheduler=None):
    """
    Load a model checkpoint.
    
    Args:
        filename (str): Checkpoint filename
        model (nn.Module): The model to load weights into
        optimizer (Optimizer, optional): The optimizer to load state into
        scheduler (LRScheduler, optional): The scheduler to load state into
    
    Returns:
        dict: Checkpoint dictionary
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        if checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


def print_epoch_summary(
    epoch,
    train_metrics,
    test_metrics,
    co_detected=False
):
    """
    Print a summary of epoch results.
    
    Args:
        epoch (int): Current epoch
        train_metrics (dict): Training metrics
        test_metrics (dict): Test metrics
        co_detected (bool): Whether CO was detected
    """
    print(f"\n{'─'*80}")
    print(f"EPOCH {epoch} SUMMARY:")
    print(f"   Train: Clean: {train_metrics['clean_acc']:5.2f}% | "
          f"Adv: {train_metrics['adv_acc']:5.2f}% | Loss: {train_metrics['loss']:.4f}")
    print(f"   Test:  Clean: {test_metrics['clean_acc']:5.2f}% | "
          f"FGSM: {test_metrics['fgsm_acc']:5.2f}% | "
          f"PGD-50: {test_metrics['pgd50_acc']:5.2f}%")
    
    if 'robustness_gap' in test_metrics:
        print(f"   Robustness Gap: {test_metrics['robustness_gap']:.2f}%")
    
    if co_detected:
        print(f"   ⚠️  CATASTROPHIC OVERFITTING DETECTED!")


def format_time(seconds):
    """
    Format seconds into a readable string.
    
    Args:
        seconds (float): Time in seconds
    
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"
