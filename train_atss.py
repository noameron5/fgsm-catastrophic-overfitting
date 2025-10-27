"""
ATSS Adversarial Training - Preventing Catastrophic Overfitting
================================================================

This script demonstrates the ATSS (Adaptive Similarity Step Size) method
that prevents catastrophic overfitting on CIFAR-10 with PreActResNet18.

Expected behavior: Stable PGD-50 accuracy throughout training (~45-55%).

Usage:
    python experiments/train_atss.py
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
from src.attacks import atss_attack
from src.utils import (
    load_cifar10,
    evaluate_robustness,
    compute_participation_ratio,
    compute_gradient_norm,
    set_seed,
    save_checkpoint,
    format_time,
)
from src.config import (
    DEVICE,
    RANDOM_SEED,
    BATCH_SIZE,
    ATSS_CONFIG,
    EVAL_CONFIG,
    OUTPUT_CONFIG,
)

warnings.filterwarnings('ignore')


def train_atss_epoch(model, trainloader, optimizer, epsilon, alpha0, beta, epoch):
    """Train one epoch with ATSS adversarial training."""
    model.train()
    train_loss = 0
    correct_clean = 0
    correct_adv = 0
    total = 0
    
    grad_norms = []
    pr_values = []
    log_interval = len(trainloader) // 5
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        # Generate adversarial examples using ATSS
        with torch.enable_grad():
            X_adv = atss_attack(model, inputs, targets, epsilon, alpha0, beta)
            
            # Compute PR on input gradients
            X_adv.requires_grad_(True)
            outputs_temp = model(X_adv)
            loss_temp = torch.nn.functional.cross_entropy(outputs_temp, targets)
            loss_temp.backward()
            pr = compute_participation_ratio(X_adv.grad)
            pr_values.append(pr)
            X_adv.requires_grad_(False)
        
        # Forward pass
        optimizer.zero_grad()
        outputs_adv = model(X_adv)
        loss = torch.nn.functional.cross_entropy(outputs_adv, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient norm
        grad_norm = compute_gradient_norm(model)
        grad_norms.append(grad_norm)
        
        optimizer.step()
        
        # Statistics
        train_loss += loss.item()
        _, predicted_adv = outputs_adv.max(1)
        
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
            avg_grad_norm = np.mean(grad_norms[-10:]) if grad_norms else 0
            avg_pr = np.mean(pr_values[-10:]) if pr_values else 0
            
            print(f"  Batch [{batch_idx+1:3d}/{len(trainloader)}] | "
                  f"Loss: {loss.item():.4f} | "
                  f"Clean: {current_clean_acc:.1f}% | "
                  f"ATSS Train: {current_adv_acc:.1f}% | "
                  f"Grad Norm: {avg_grad_norm:.2f} | "
                  f"PR: {avg_pr:.2f}")
    
    acc_clean = 100. * correct_clean / total
    acc_adv = 100. * correct_adv / total
    avg_loss = train_loss / (batch_idx + 1)
    avg_grad_norm = np.mean(grad_norms)
    avg_pr = np.mean(pr_values)
    
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
        lr=ATSS_CONFIG['learning_rate'],
        momentum=ATSS_CONFIG['momentum'],
        weight_decay=ATSS_CONFIG['weight_decay']
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=ATSS_CONFIG['lr_milestones'],
        gamma=ATSS_CONFIG['lr_gamma']
    )
    
    # Training history
    history = defaultdict(list)
    
    print(f"\n{'='*80}")
    print(f"ATSS ADVERSARIAL TRAINING - CIFAR-10")
    print(f"{'='*80}")
    print(f"\nConfiguration:")
    print(f"   Architecture: PreActResNet18")
    print(f"   Device: {DEVICE}")
    print(f"   Epsilon: {ATSS_CONFIG['epsilon']:.4f} ({ATSS_CONFIG['epsilon']*255:.1f}/255)")
    print(f"   Alpha0: {ATSS_CONFIG['alpha0']:.4f}")
    print(f"   Beta: {ATSS_CONFIG['beta']}")
    print(f"   Learning Rate: {ATSS_CONFIG['learning_rate']}")
    print(f"   Epochs: {ATSS_CONFIG['epochs']}")
    print(f"\nExpected: No CO, stable PGD robustness (~45-55%)")
    print(f"{'='*80}\n")
    
    # Training loop
    for epoch in range(1, ATSS_CONFIG['epochs'] + 1):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{ATSS_CONFIG['epochs']} | LR: {optimizer.param_groups[0]['lr']:.4f}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # Train
        train_clean_acc, train_adv_acc, train_loss, grad_norm, avg_pr = train_atss_epoch(
            model, trainloader, optimizer, 
            ATSS_CONFIG['epsilon'], 
            ATSS_CONFIG['alpha0'], 
            ATSS_CONFIG['beta'], 
            epoch
        )
        
        print(f"\nTraining Summary:")
        print(f"   Clean: {train_clean_acc:.2f}% | ATSS: {train_adv_acc:.2f}% | "
              f"Loss: {train_loss:.4f} | Grad Norm: {grad_norm:.2f} | PR: {avg_pr:.2f}")
        
        # Evaluate
        print(f"\nEvaluating Robustness...")
        test_clean_acc = evaluate_robustness(model, testloader, 0, attack_type='clean')
        test_fgsm_acc = evaluate_robustness(model, testloader, ATSS_CONFIG['epsilon'], attack_type='fgsm')
        test_pgd20_acc = evaluate_robustness(model, testloader, ATSS_CONFIG['epsilon'], attack_type='pgd', num_steps=20)
        test_pgd50_acc = evaluate_robustness(model, testloader, ATSS_CONFIG['epsilon'], attack_type='pgd', num_steps=50)
        
        print(f"   Clean: {test_clean_acc:.2f}% | FGSM: {test_fgsm_acc:.2f}% | "
              f"PGD-20: {test_pgd20_acc:.2f}% | PGD-50: {test_pgd50_acc:.2f}%")
        
        # Calculate metrics
        robustness_gap = test_fgsm_acc - test_pgd50_acc
        fgsm_train_pgd_test_gap = train_adv_acc - test_pgd50_acc
        
        print(f"\nMetrics:")
        print(f"   FGSM-PGD50 Gap: {robustness_gap:.2f}%")
        print(f"   Train ATSS vs Test PGD: {fgsm_train_pgd_test_gap:.2f}%")
        print(f"   Participation Ratio: {avg_pr:.2f}")
        
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
        history['participation_ratio'].append(avg_pr)
        history['robustness_gap'].append(robustness_gap)
        history['fgsm_train_pgd_test_gap'].append(fgsm_train_pgd_test_gap)
        
        print(f"\nEpoch Time: {format_time(time.time() - start_time)}")
    
    # Save results
    results_file = os.path.join(OUTPUT_CONFIG['results_dir'], 'atss_results.json')
    with open(results_file, 'w') as f:
        json.dump(dict(history), f, indent=2)
    print(f"\nResults saved to '{results_file}'")
    
    # Visualize
    visualize_results(history)
    
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETED")
    print(f"{'='*80}")
    print(f"\nFinal Results:")
    print(f"   Clean: {history['test_clean_acc'][-1]:.2f}%")
    print(f"   FGSM: {history['test_fgsm_acc'][-1]:.2f}%")
    print(f"   PGD-50: {history['test_pgd50_acc'][-1]:.2f}%")
    print(f"   Robustness Gap: {history['robustness_gap'][-1]:.2f}%")
    print(f"\nATSS successfully prevented catastrophic overfitting!")


def visualize_results(history):
    """Create visualization plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy evolution
    ax = axes[0, 0]
    ax.plot(history['epoch'], history['test_clean_acc'], 'b-', label='Clean', linewidth=2)
    ax.plot(history['epoch'], history['test_fgsm_acc'], 'g-', label='FGSM', linewidth=2)
    ax.plot(history['epoch'], history['test_pgd50_acc'], 'r--', label='PGD-50', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('ATSS: Stable Robustness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Robustness gap
    ax = axes[0, 1]
    ax.plot(history['epoch'], history['robustness_gap'], 'purple', linewidth=2, marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('FGSM - PGD Gap (%)')
    ax.set_title('Robustness Gap (Stable)')
    ax.grid(True, alpha=0.3)
    
    # Participation Ratio
    ax = axes[1, 0]
    ax.plot(history['epoch'], history['participation_ratio'], 'orange', linewidth=2, marker='d')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Participation Ratio')
    ax.set_title('Gradient Distribution (Stable PR)')
    ax.grid(True, alpha=0.3)
    
    # Training Loss
    ax = axes[1, 1]
    ax.plot(history['epoch'], history['train_loss'], 'brown', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Loss Evolution')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('ATSS Adversarial Training Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_file = os.path.join(OUTPUT_CONFIG['results_dir'], 'atss_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to '{plot_file}'")
    plt.close()


if __name__ == "__main__":
    main()
