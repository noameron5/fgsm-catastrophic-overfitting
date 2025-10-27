"""
Catastrophic Overfitting Demo Package
======================================

A comprehensive implementation demonstrating catastrophic overfitting
in fast adversarial training and its prevention using ATSS.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from src.models import PreActResNet18
from src.attacks import fgsm_attack, pgd_attack, atss_attack
from src.utils import load_cifar10, evaluate_robustness, compute_participation_ratio

__all__ = [
    "PreActResNet18",
    "fgsm_attack",
    "pgd_attack",
    "atss_attack",
    "load_cifar10",
    "evaluate_robustness",
    "compute_participation_ratio",
]
