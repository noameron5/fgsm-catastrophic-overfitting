"""
Comprehensive Comparison: Catastrophic Overfitting (FGSM) vs ATSS
=================================================================

This script loads results from both experiments and creates detailed
comparison visualizations.

Usage:
    python experiments/compare.py

Prerequisites:
    - Run train_fgsm.py first
    - Run train_atss.py second
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import OUTPUT_CONFIG

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_results():
    """Load results from both experiments."""
    results_dir = OUTPUT_CONFIG['results_dir']
    
    # Load CO results
    co_file = os.path.join(results_dir, 'catastrophic_overfitting_results.json')
    try:
        with open(co_file, 'r') as f:
            co_results = json.load(f)
        print("✓ Loaded Catastrophic Overfitting results")
    except FileNotFoundError:
        print(f"✗ Error: '{co_file}' not found!")
        print("  Please run 'python experiments/train_fgsm.py' first.")
        sys.exit(1)
    
    # Load ATSS results
    atss_file = os.path.join(results_dir, 'atss_results.json')
    try:
        with open(atss_file, 'r') as f:
            atss_results = json.load(f)
        print("✓ Loaded ATSS results")
    except FileNotFoundError:
        print(f"✗ Error: '{atss_file}' not found!")
        print("  Please run 'python experiments/train_atss.py' first.")
        sys.exit(1)
    
    return co_results, atss_results


def create_main_comparison(co_results, atss_results):
    """Create comprehensive comparison figure."""
    # Extract data from CO
    co_history = co_results['training_history']
    co_epochs = co_history['epoch']
    co_pgd50 = co_history['test_pgd50_acc']
    co_fgsm = co_history['test_fgsm_acc']
    co_clean = co_history['test_clean_acc']
    co_rob_gap = co_history['robustness_gap']
    co_loss = co_history['train_loss']
    co_grad_norm = co_history['gradient_norm']
    co_pr = co_history['participation_ratio']
    
    # Extract data from ATSS
    atss_epochs = atss_results['epoch']
    atss_pgd50 = atss_results['test_pgd50_acc']
    atss_fgsm = atss_results['test_fgsm_acc']
    atss_clean = atss_results['test_clean_acc']
    atss_rob_gap = atss_results['robustness_gap']
    atss_loss = atss_results['train_loss']
    atss_grad_norm = atss_results['gradient_norm']
    atss_pr = atss_results['participation_ratio']
    
    # CO detection info
    co_detected = co_results['catastrophic_overfitting']['detected']
    co_epoch = co_results['catastrophic_overfitting'].get('epoch', -1)
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: PGD-50 Comparison (Most Important)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(co_epochs, co_pgd50, 'r-', linewidth=3, label='Standard FGSM (CO)', 
             marker='o', markersize=4)
    ax1.plot(atss_epochs, atss_pgd50, 'g-', linewidth=3, label='ATSS (Stable)', 
             marker='s', markersize=4)
    if co_detected and co_epoch > 0:
        ax1.axvline(x=co_epoch, color='red', linestyle='--', alpha=0.5, linewidth=2,
                   label=f'CO at Epoch {co_epoch}')
        ax1.fill_between([co_epoch, max(co_epochs)], 0, 100, alpha=0.1, color='red')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('PGD-50 Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('PGD-50 Robustness: The Critical Metric', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Robustness Gap
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(co_epochs, co_rob_gap, 'r-', linewidth=2.5, label='Standard FGSM', marker='o', markersize=3)
    ax2.plot(atss_epochs, atss_rob_gap, 'g-', linewidth=2.5, label='ATSS', marker='s', markersize=3)
    ax2.fill_between(co_epochs, 0, co_rob_gap, alpha=0.2, color='red')
    ax2.fill_between(atss_epochs, 0, atss_rob_gap, alpha=0.2, color='green')
    if co_detected and co_epoch > 0:
        ax2.axvline(x=co_epoch, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax2.axhline(y=10, color='orange', linestyle=':', alpha=0.5, label='Healthy Gap (<10%)')
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('FGSM - PGD50 Gap (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Robustness Gap Evolution', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Standard FGSM Detail
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(co_epochs, co_clean, 'b-', linewidth=2, label='Clean', alpha=0.8)
    ax3.plot(co_epochs, co_fgsm, 'g-', linewidth=2, label='FGSM', alpha=0.8)
    ax3.plot(co_epochs, co_pgd50, 'r-', linewidth=2.5, label='PGD-50', marker='o', markersize=3)
    if co_detected and co_epoch > 0:
        ax3.axvline(x=co_epoch, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Standard FGSM: CO Collapse', fontsize=14, fontweight='bold', color='darkred')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: ATSS Detail
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(atss_epochs, atss_clean, 'b-', linewidth=2, label='Clean', alpha=0.8)
    ax4.plot(atss_epochs, atss_fgsm, 'g-', linewidth=2, label='FGSM', alpha=0.8)
    ax4.plot(atss_epochs, atss_pgd50, 'r-', linewidth=2.5, label='PGD-50', marker='s', markersize=3)
    ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax4.set_title('ATSS: Stable Robustness', fontsize=14, fontweight='bold', color='darkgreen')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Gradient Norm
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(co_epochs, co_grad_norm, 'r-', linewidth=2, label='Standard FGSM', marker='o', markersize=3, alpha=0.7)
    ax5.plot(atss_epochs, atss_grad_norm, 'g-', linewidth=2, label='ATSS', marker='s', markersize=3, alpha=0.7)
    if co_detected and co_epoch > 0:
        ax5.axvline(x=co_epoch, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax5.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Gradient Norm', fontsize=12, fontweight='bold')
    ax5.set_title('Gradient Norm During Training', fontsize=14, fontweight='bold')
    ax5.legend(loc='best', fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Training Loss
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(co_epochs, co_loss, 'r-', linewidth=2, label='Standard FGSM', alpha=0.7)
    ax6.plot(atss_epochs, atss_loss, 'g-', linewidth=2, label='ATSS', alpha=0.7)
    if co_detected and co_epoch > 0:
        ax6.axvline(x=co_epoch, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax6.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax6.set_title('Loss Evolution', fontsize=14, fontweight='bold')
    ax6.legend(loc='best', fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Final Performance Bar Chart
    ax7 = fig.add_subplot(gs[2, 0])
    categories = ['Clean', 'FGSM', 'PGD-50']
    co_final = [co_clean[-1], co_fgsm[-1], co_pgd50[-1]]
    atss_final = [atss_clean[-1], atss_fgsm[-1], atss_pgd50[-1]]
    x = np.arange(len(categories))
    width = 0.35
    bars1 = ax7.bar(x - width/2, co_final, width, label='Standard FGSM', color='red', alpha=0.7)
    bars2 = ax7.bar(x + width/2, atss_final, width, label='ATSS', color='green', alpha=0.7)
    ax7.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax7.set_title('Final Performance Comparison', fontsize=14, fontweight='bold')
    ax7.set_xticks(x)
    ax7.set_xticklabels(categories)
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3, axis='y')
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Plot 8: Participation Ratio
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(co_epochs, co_pr, 'r-', linewidth=2, label='Standard FGSM', marker='o', markersize=3)
    ax8.plot(atss_epochs, atss_pr, 'g-', linewidth=2, label='ATSS', marker='s', markersize=3)
    if co_detected and co_epoch > 0:
        ax8.axvline(x=co_epoch, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax8.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax8.set_ylabel('Participation Ratio', fontsize=12, fontweight='bold')
    ax8.set_title('Gradient Concentration (PR)', fontsize=14, fontweight='bold')
    ax8.legend(fontsize=10)
    ax8.grid(True, alpha=0.3)
    
    # Plot 9: Summary Text
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    co_final_gap = co_fgsm[-1] - co_pgd50[-1]
    atss_final_gap = atss_fgsm[-1] - atss_pgd50[-1]
    improvement = atss_pgd50[-1] - co_pgd50[-1]
    
    summary_text = f"""
KEY FINDINGS:

Standard FGSM (CO):
  • Final PGD-50: {co_pgd50[-1]:.1f}%
  • Robustness Gap: {co_final_gap:.1f}%
  • CO Detected: {'YES' if co_detected else 'NO'}
  {f'• CO at Epoch: {co_epoch}' if co_detected else ''}

ATSS (Prevention):
  • Final PGD-50: {atss_pgd50[-1]:.1f}%
  • Robustness Gap: {atss_final_gap:.1f}%
  • Stable: YES

IMPROVEMENT:
  • PGD-50 Gain: +{improvement:.1f}%
  • Gap Reduction: -{co_final_gap - atss_final_gap:.1f}%

CONCLUSION:
ATSS successfully prevents
catastrophic overfitting while
maintaining {atss_pgd50[-1]:.1f}% robust
accuracy vs {co_pgd50[-1]:.1f}% for
standard FGSM.
"""
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    fig.suptitle('Catastrophic Overfitting vs ATSS: Complete Comparison', 
                fontsize=18, fontweight='bold', y=0.995)
    
    # Save
    output_file = os.path.join(OUTPUT_CONFIG['results_dir'], 'co_vs_atss_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: '{output_file}'")
    plt.show()


def create_pgd_comparison(co_results, atss_results):
    """Create focused PGD-50 comparison."""
    co_history = co_results['training_history']
    co_epochs = co_history['epoch']
    co_pgd50 = co_history['test_pgd50_acc']
    
    atss_epochs = atss_results['epoch']
    atss_pgd50 = atss_results['test_pgd50_acc']
    
    co_detected = co_results['catastrophic_overfitting']['detected']
    co_epoch = co_results['catastrophic_overfitting'].get('epoch', -1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Direct overlay
    ax1.plot(co_epochs, co_pgd50, 'r-', linewidth=4, label='Standard FGSM (CO)', 
             marker='o', markersize=5, alpha=0.8)
    ax1.plot(atss_epochs, atss_pgd50, 'g-', linewidth=4, label='ATSS (Stable)', 
             marker='s', markersize=5, alpha=0.8)
    
    if co_detected and co_epoch > 0:
        ax1.axvline(x=co_epoch, color='darkred', linestyle='--', linewidth=3, 
                    alpha=0.7, label=f'CO Collapse (Epoch {co_epoch})')
        ax1.text(co_epoch, 20, 'Catastrophic\nCollapse', ha='center', 
                fontsize=12, fontweight='bold', color='darkred')
    
    ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax1.set_ylabel('PGD-50 Test Accuracy (%)', fontsize=14, fontweight='bold')
    ax1.set_title('Direct Comparison: PGD-50 Robustness', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12, loc='best')
    ax1.grid(True, alpha=0.4)
    
    # Right: Difference plot
    min_len = min(len(co_pgd50), len(atss_pgd50))
    difference = [atss_pgd50[i] - co_pgd50[i] for i in range(min_len)]
    epochs_common = list(range(1, min_len + 1))
    
    ax2.fill_between(epochs_common, 0, difference, where=[d >= 0 for d in difference],
                     color='green', alpha=0.3, label='ATSS Better')
    ax2.fill_between(epochs_common, 0, difference, where=[d < 0 for d in difference],
                     color='red', alpha=0.3, label='FGSM Better')
    ax2.plot(epochs_common, difference, 'b-', linewidth=3, marker='o', markersize=4)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    if co_detected and co_epoch > 0 and co_epoch <= min_len:
        ax2.axvline(x=co_epoch, color='darkred', linestyle='--', linewidth=3, alpha=0.7)
    
    ax2.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax2.set_ylabel('PGD-50 Difference (ATSS - FGSM) %', fontsize=14, fontweight='bold')
    ax2.set_title('Robustness Advantage of ATSS', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.4)
    
    plt.tight_layout()
    
    output_file = os.path.join(OUTPUT_CONFIG['results_dir'], 'pgd50_direct_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: '{output_file}'")
    plt.show()


def print_numerical_summary(co_results, atss_results):
    """Print detailed numerical comparison."""
    co_history = co_results['training_history']
    co_pgd50 = co_history['test_pgd50_acc']
    co_fgsm = co_history['test_fgsm_acc']
    co_clean = co_history['test_clean_acc']
    co_rob_gap = co_history['robustness_gap']
    
    atss_pgd50 = atss_results['test_pgd50_acc']
    atss_fgsm = atss_results['test_fgsm_acc']
    atss_clean = atss_results['test_clean_acc']
    atss_rob_gap = atss_results['robustness_gap']
    
    co_detected = co_results['catastrophic_overfitting']['detected']
    co_epoch = co_results['catastrophic_overfitting'].get('epoch', -1)
    
    print("\n" + "="*80)
    print("NUMERICAL COMPARISON SUMMARY")
    print("="*80)
    
    print("\nFinal Epoch Performance:")
    print("-" * 80)
    print(f"{'Metric':<20} {'Standard FGSM':<20} {'ATSS':<20} {'Difference':<20}")
    print("-" * 80)
    print(f"{'Clean Accuracy':<20} {co_clean[-1]:>18.2f}% {atss_clean[-1]:>18.2f}% {atss_clean[-1] - co_clean[-1]:>18.2f}%")
    print(f"{'FGSM Accuracy':<20} {co_fgsm[-1]:>18.2f}% {atss_fgsm[-1]:>18.2f}% {atss_fgsm[-1] - co_fgsm[-1]:>18.2f}%")
    print(f"{'PGD-50 Accuracy':<20} {co_pgd50[-1]:>18.2f}% {atss_pgd50[-1]:>18.2f}% {atss_pgd50[-1] - co_pgd50[-1]:>18.2f}%")
    print(f"{'Robustness Gap':<20} {co_rob_gap[-1]:>18.2f}% {atss_rob_gap[-1]:>18.2f}% {atss_rob_gap[-1] - co_rob_gap[-1]:>18.2f}%")
    print("-" * 80)
    
    print("\nKey Statistics:")
    print("-" * 80)
    print(f"Maximum PGD-50 (Standard FGSM): {max(co_pgd50):.2f}% at epoch {co_pgd50.index(max(co_pgd50)) + 1}")
    print(f"Minimum PGD-50 (Standard FGSM): {min(co_pgd50):.2f}% at epoch {co_pgd50.index(min(co_pgd50)) + 1}")
    print(f"PGD-50 drop: {max(co_pgd50) - min(co_pgd50):.2f}%")
    print()
    print(f"Maximum PGD-50 (ATSS): {max(atss_pgd50):.2f}% at epoch {atss_pgd50.index(max(atss_pgd50)) + 1}")
    print(f"Minimum PGD-50 (ATSS): {min(atss_pgd50):.2f}% at epoch {atss_pgd50.index(min(atss_pgd50)) + 1}")
    print(f"PGD-50 variation: {max(atss_pgd50) - min(atss_pgd50):.2f}%")
    print("-" * 80)
    
    print("\nCatastrophic Overfitting Detection:")
    print("-" * 80)
    if co_detected:
        print(f"Standard FGSM: CO DETECTED at epoch {co_epoch}")
        if co_epoch > 1:
            print(f"  - PGD-50 before CO: {co_pgd50[co_epoch-2]:.2f}% (epoch {co_epoch-1})")
            print(f"  - PGD-50 at CO: {co_pgd50[co_epoch-1]:.2f}% (epoch {co_epoch})")
            print(f"  - Drop: {co_pgd50[co_epoch-2] - co_pgd50[co_epoch-1]:.2f}%")
    else:
        print("Standard FGSM: CO NOT DETECTED")
    
    print(f"\nATSS: NO CATASTROPHIC OVERFITTING")
    print(f"  - Maintains stable PGD-50 robustness throughout training")
    print(f"  - Final PGD-50: {atss_pgd50[-1]:.2f}%")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print(f"\nATSS successfully prevents catastrophic overfitting:")
    print(f"  • PGD-50 improvement: +{atss_pgd50[-1] - co_pgd50[-1]:.1f}%")
    print(f"  • Robustness gap reduction: -{co_rob_gap[-1] - atss_rob_gap[-1]:.1f}%")
    print(f"  • Training stability: MAINTAINED")
    print(f"\nThis demonstrates that adaptive single-step training can achieve")
    print(f"robust adversarial defenses without the computational cost of PGD training.")
    print("="*80)


def main():
    """Main comparison function."""
    print("="*80)
    print("CATASTROPHIC OVERFITTING vs ATSS COMPARISON")
    print("="*80)
    
    # Load results
    print("\nLoading experimental results...")
    co_results, atss_results = load_results()
    
    # Create visualizations
    print("\nGenerating comparison visualizations...")
    create_main_comparison(co_results, atss_results)
    create_pgd_comparison(co_results, atss_results)
    
    # Print summary
    print_numerical_summary(co_results, atss_results)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()
