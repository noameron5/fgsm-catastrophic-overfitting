"""
Configuration file for catastrophic overfitting experiments.

This file contains all hyperparameters and settings used in the experiments.
Modify these values to run different experimental configurations.
"""

import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Random seed for reproducibility
RANDOM_SEED = 42

# Data configuration
BATCH_SIZE = 128
NUM_WORKERS = 2
DATA_DIR = './data'

# Training configuration - FGSM (Catastrophic Overfitting)
FGSM_CONFIG = {
    'epsilon': 10/255,              # Perturbation budget (above 8/255 threshold triggers CO)
    'learning_rate': 0.1,           # Initial learning rate
    'epochs': 30,                   # Total training epochs
    'warmup_epochs': 3,             # Warmup period with reduced epsilon
    'use_rs_fgsm_early': True,      # Use random start FGSM initially
    'momentum': 0.9,                # SGD momentum
    'weight_decay': 5e-4,           # L2 regularization
    'lr_milestones': [10, 18, 25],  # Learning rate reduction epochs
    'lr_gamma': 0.5,                # Learning rate reduction factor
}

# Training configuration - ATSS (Prevention)
ATSS_CONFIG = {
    'epsilon': 10/255,              # Perturbation budget
    'alpha0': 10/255,               # Base step size (typically equal to epsilon)
    'beta': 0.5,                    # Influence coefficient for adaptive step size
    'learning_rate': 0.1,           # Initial learning rate
    'epochs': 30,                   # Total training epochs
    'momentum': 0.9,                # SGD momentum
    'weight_decay': 5e-4,           # L2 regularization
    'lr_milestones': [10, 18, 25],  # Learning rate reduction epochs
    'lr_gamma': 0.5,                # Learning rate reduction factor
}

# Evaluation configuration
EVAL_CONFIG = {
    'pgd_steps': 50,                # Number of PGD steps for robust evaluation
    'pgd_alpha': None,              # PGD step size (None = epsilon/4)
    'pgd_restarts': 1,              # Number of random restarts for PGD
    'eval_frequency': 1,            # Evaluate every N epochs
    'multi_epsilon_values': [0, 2/255, 4/255, 8/255, 12/255, 16/255],
}

# Logging configuration
LOG_CONFIG = {
    'verbose': True,                # Print detailed logs
    'log_interval_batches': 5,      # Log N times per epoch
    'save_checkpoints': True,       # Save model checkpoints
    'save_co_checkpoint': True,     # Save checkpoint when CO detected
}

# Output directories
OUTPUT_CONFIG = {
    'results_dir': 'results',
    'checkpoint_dir': 'results/checkpoints',
    'plots_dir': 'results/plots',
}

# Catastrophic Overfitting detection thresholds
CO_DETECTION = {
    'robustness_gap_threshold': 90,     # Gap > 90% indicates CO
    'min_pgd_accuracy': 5,              # PGD accuracy < 5% indicates CO
    'fgsm_train_pgd_test_gap': 60,      # Warning threshold
}

# Model configuration
MODEL_CONFIG = {
    'architecture': 'PreActResNet18',
    'num_classes': 10,                  # CIFAR-10
}

# Clamp values for image pixels
CLAMP_MIN = 0
CLAMP_MAX = 1
