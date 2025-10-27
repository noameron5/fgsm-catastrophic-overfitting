"""
Catastrophic Overfitting Demonstration with FGSM Adversarial Training
=====================================================================

This script demonstrates catastrophic overfitting on CIFAR-10 with PreActResNet18.
Expected behavior: FGSM accuracy ~100%, PGD-50 accuracy ~0% after CO onset.

Usage:
    python experiments/train_fgsm.py
"""

import os
import sys
import json
import time
import warnings
from collections import defaultdict

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import PreActResNet18
from src.attacks import fgsm_attack, pgd_attack
from src.utils import (
    load_cifar10,
    evaluate_robustness,
    compute_participation_ratio,
    compute_gradient_norm,
    detect_catastrophic_overfitting,
    set_seed,
    save_checkpoint,
    format_time,
)
from src.config import (
    DEVICE,
    RANDOM_SEED,
    BATCH_SIZE,
    FGSM_CONFIG,
    EVAL_CONFIG,
    OUTPUT_CONFIG,
    CO_DETECTION,
)

warnings.filterwarnings('ignore')


def train_fgsm_epoch(model, trainloader, optimizer, epsilon, epoch, use_random_start=False):
    """Train one epoch with FGSM adversarial training."""
    model.train()
    train_loss = 0
    correct_clean = 0
    correct_adv = 0
    total = 0
    
    batch_losses = []
    grad_norms = []
    batch_prs = []
    log_interval = len(trainloader) // 5
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        # Generate adversarial examples
        with torch.enable_grad():
            X_adv = fgsm_attack(model, inputs, targets, epsilon, random_start=use_random_start)
        
        # Forward pass on adversarial examples
        optimizer.zero_grad()
        outputs_adv = model(X_adv)
        loss = torch.nn.functional.cross_entropy(outputs_adv, targets)
        
        # Backward pass
        loss.backward()
        
        # Calculate gradient norm
        grad_norm = compute_gradient_norm(model)
        grad_norms.append(grad_norm)
        
        # Calculate Participation Ratio on input gradients
        inputs_pr = inputs.clone().detach().requires_grad_(True)
        outputs_pr = model(inputs_pr)
        loss_pr = torch.nn.functional.cross_entropy(outputs_pr, targets)
        loss_pr.backward()
        
        if inputs_pr.grad is not None:
            pr_value = compute_participation_ratio(inputs_pr.grad)
            batch_prs.append(pr_value)
        
        optimizer.step()
        
        # Track statistics
        train_loss += loss.item()
        batch_losses.append(loss.item())
        _, predicted_adv = outputs_adv.max(1)
        
        # Clean accuracy
        with torch.no_grad():
            outputs_clean = model(inputs)
            _, predicted_clean = outputs_clean.max(1)
        
        total += targets.size(0)
        correct_clean += predicted_clean.eq(targets).sum().item()
        correct_adv += predicted_adv.eq(targets).sum().item()
        
        # Batch logging
        if batch_idx % log_interval == 0 or batch_idx == len(trainloader) - 1:
            current_clean_acc = 100. * correct_clean / total
            current_adv_acc = 100. * correct_adv / total
            avg_grad_norm = np.mean(grad_norms[-10:]) if len(grad_norms) > 0 else 0
            avg_pr = np.mean(batch_prs[-10:]) if len(batch_prs) > 0 else 0
            
            print(f"  Batch [{batch_idx+1:3d}/{len(trainloader)}] | "
                  f"Loss: {loss.item():.4f} | "
                  f"Clean: {current_clean_acc:.1f}% | "
                  f"FGSM Train: {current_adv_acc:.1f}% | "
                  f"Grad Norm: {avg_grad_norm:.2f} | "
                  f"PR: {avg_pr:.2f}")
    
    acc_clean = 100. * correct_clean / total
    acc_adv = 100. * correct_adv / total
    avg_loss = train_loss / (batch_idx + 1)
    avg_grad_norm = np.mean(grad_norms)
    avg_pr = np.mean(batch_prs) if batch_prs else 0
    
    return acc_clean, acc_adv, avg_loss, avg_grad_norm, avg_pr


def main():
    """Main training function."""
    # Setup
    set_seed(RANDOM_SEED)
    os.makedirs(OUTPUT_CONFIG['results_dir'], exist_ok=True)
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    trainloader, testloader = load_cifar10(batch_size=BATCH_SIZE)
    print(f"Training samples: {len(trainloader.dataset)}")
    print(f"Test samples: {len(testloader.dataset)}")
    
    # Initialize model
    model = PreActResNet18().to(DEVICE)
    optimizer = optim.SGD(
        model.parameters(),
        lr=FGSM_CONFIG['learning_rate'],
        momentum=FGSM_CONFIG['momentum'],
        weight_decay=FGSM_CONFIG['weight_decay']
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=FGSM_CONFIG['lr_milestones'],
        gamma=FGSM_CONFIG['lr_gamma']
    )
    
    # Training history
    history = defaultdict(list)
    co_detected = False
    co_epoch = -1
    participation_ratios = []
    
    print(f"\n{'='*80}")
    print(f"CATASTROPHIC OVERFITTING EXPERIMENT - CIFAR-10")
    print(f"{'='*80}")
    print(f"\nConfiguration:")
    print(f"   Architecture: PreActResNet18")
    print(f"   Device: {DEVICE}")
    print(f"   Epsilon: {FGSM_CONFIG['epsilon']:.4f} ({FGSM_CONFIG['epsilon']*255:.1f}/255)")
    print(f"   Learning Rate: {FGSM_CONFIG['learning_rate']}")
    print(f"   Epochs: {FGSM_CONFIG['epochs']}")
    print(f"   Warmup Epochs: {FGSM_CONFIG['warmup_epochs']}")
    print(f"\nExpected: CO around epochs 8-12, FGSM ~100%, PGD-50 ~0%")
    print(f"{'='*80}\n")
    
    # Training loop
    for epoch in range(1, FGSM_CONFIG['epochs'] + 1):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{FGSM_CONFIG['epochs']} | LR: {optimizer.param_groups[0]['lr']:.4f}")
        print(f"{'='*80}")
        
        # Adaptive epsilon and random start strategy
        if epoch <= FGSM_CONFIG['warmup_epochs']:
            current_epsilon = (6/255) + (epoch-1) * ((8/255 - 6/255) / FGSM_CONFIG['warmup_epochs'])
            use_random_start = True
        elif epoch <= 7:
            current_epsilon = 8/255
            use_random_start = True
        elif epoch == 8:
            current_epsilon = FGSM_CONFIG['epsilon']
            use_random_start = False
        elif epoch >= 9 and epoch <= 11:
            current_epsilon = FGSM_CONFIG['epsilon'] + (epoch - 8) * (1/255)
            use_random_start = False
        else:
            current_epsilon = FGSM_CONFIG['epsilon'] + (3/255)
            use_random_start = False
        
        start_time = time.time()
        
        # Train
        train_clean_acc, train_adv_acc, train_loss, grad_norm, pr_value = train_fgsm_epoch(
            model, trainloader, optimizer, current_epsilon, epoch, use_random_start
        )
        
        print(f"\nTraining Summary:")
        print(f"   Train Clean: {train_clean_acc:.2f}% | Train FGSM: {train_adv_acc:.2f}% | "
              f"Loss: {train_loss:.4f} | Grad Norm: {grad_norm:.2f} | PR: {pr_value:.2f}")
        
        # Evaluate
        print(f"\nEvaluating Robustness...")
        test_clean_acc = evaluate_robustness(model, testloader, 0, attack_type='clean')
        test_fgsm_acc = evaluate_robustness(model, testloader, current_epsilon, attack_type='fgsm')
        test_pgd20_acc = evaluate_robustness(model, testloader, current_epsilon, attack_type='pgd', num_steps=20)
        
        if epoch % 2 == 0 or epoch > 6:
            test_pgd50_acc = evaluate_robustness(model, testloader, current_epsilon, attack_type='pgd', num_steps=50)
        else:
            test_pgd50_acc = test_pgd20_acc
        
        print(f"   Clean: {test_clean_acc:.2f}% | FGSM: {test_fgsm_acc:.2f}% | "
              f"PGD-20: {test_pgd20_acc:.2f}% | PGD-50: {test_pgd50_acc:.2f}%")
        
        # Calculate metrics
        robustness_gap = test_fgsm_acc - test_pgd50_acc
        fgsm_train_pgd_test_gap = train_adv_acc - test_pgd50_acc
        
        print(f"\nKey CO Metrics:")
        print(f"   FGSM Train: {train_adv_acc:.2f}% | PGD-50 Test: {test_pgd50_acc:.2f}% | "
              f"Gap: {fgsm_train_pgd_test_gap:.2f}%")
        print(f"   Participation Ratio: {pr_value:.2f}")
        
        # Check for CO
        if not co_detected and robustness_gap > CO_DETECTION['robustness_gap_threshold'] and \
           test_pgd50_acc < CO_DETECTION['min_pgd_accuracy']:
            co_detected = True
            co_epoch = epoch
            print(f"\n{'='*80}")
            print(f"CATASTROPHIC OVERFITTING DETECTED at epoch {epoch}!")
            print(f"{'='*80}")
            print(f"   FGSM Test: {test_fgsm_acc:.2f}% | PGD-50 Test: {test_pgd50_acc:.2f}%")
            print(f"   Gap: {fgsm_train_pgd_test_gap:.2f}% | PR: {pr_value:.2f}")
            print(f"{'='*80}\n")
        
        # Update scheduler
        scheduler.step()
        
        # Store history
        history['epoch'].append(epoch)
        history['train_clean_acc'].append(train_clean_acc)
        history['train_adv_acc'].append(train_adv_acc)
        history['test_clean_acc'].append(test_clean_acc)
        history['test_fgsm_acc'].append(test_fgsm_acc)
        history['test_pgd20_acc'].append(test_pgd20_acc)
        history['test_pgd50_acc'].append(test_pgd50_acc)
        history['train_loss'].append(train_loss)
        history['gradient_norm'].append(grad_norm)
        history['robustness_gap'].append(robustness_gap)
        history['fgsm_train_pgd_test_gap'].append(fgsm_train_pgd_test_gap)
        history['participation_ratio'].append(pr_value)
        participation_ratios.append(pr_value)
        
        print(f"\nEpoch Time: {format_time(time.time() - start_time)}")
    
    # Save results
    results = {
        'experiment_info': {
            'title': 'Catastrophic Overfitting in FGSM Adversarial Training',
            'dataset': 'CIFAR-10',
            'architecture': 'PreActResNet18',
        },
        'catastrophic_overfitting': {
            'detected': co_detected,
            'epoch': co_epoch if co_detected else None,
            'min_pr': float(min(participation_ratios)),
            'min_pr_epoch': participation_ratios.index(min(participation_ratios)) + 1,
        },
        'training_history': dict(history),
    }
    
    results_file = os.path.join(OUTPUT_CONFIG['results_dir'], 'catastrophic_overfitting_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to '{results_file}'")
    
    # Visualize
    visualize_results(history, co_detected, co_epoch, participation_ratios)
    
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETED")
    print(f"{'='*80}")


def visualize_results(history, co_detected, co_epoch, participation_ratios):
    """Create visualization plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy evolution
    ax = axes[0, 0]
    ax.plot(history['epoch'], history['test_clean_acc'], 'b-', label='Clean', linewidth=2)
    ax.plot(history['epoch'], history['test_fgsm_acc'], 'g-', label='FGSM', linewidth=2)
    ax.plot(history['epoch'], history['test_pgd50_acc'], 'r--', label='PGD-50', linewidth=2)
    if co_detected:
        ax.axvline(x=co_epoch, color='orange', linestyle='--', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Catastrophic Overfitting: Accuracy Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Robustness gap
    ax = axes[0, 1]
    ax.plot(history['epoch'], history['robustness_gap'], 'purple', linewidth=2, marker='o')
    if co_detected:
        ax.axvline(x=co_epoch, color='orange', linestyle='--', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('FGSM - PGD Gap (%)')
    ax.set_title('Robustness Gap')
    ax.grid(True, alpha=0.3)
    
    # Participation Ratio
    ax = axes[1, 0]
    ax.plot(history['epoch'], participation_ratios, 'darkblue', linewidth=2, marker='o')
    if co_detected:
        ax.axvline(x=co_epoch, color='orange', linestyle='--', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Participation Ratio')
    ax.set_title('Gradient Concentration (PR)')
    ax.grid(True, alpha=0.3)
    
    # Train vs Test
    ax = axes[1, 1]
    ax.plot(history['epoch'], history['train_adv_acc'], 'g-', label='FGSM Train', linewidth=2)
    ax.plot(history['epoch'], history['test_pgd50_acc'], 'r-', label='PGD-50 Test', linewidth=2)
    if co_detected:
        ax.axvline(x=co_epoch, color='orange', linestyle='--', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Training vs Test Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Catastrophic Overfitting Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_file = os.path.join(OUTPUT_CONFIG['results_dir'], 'co_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to '{plot_file}'")
    plt.close()


if __name__ == "__main__":
    main()
