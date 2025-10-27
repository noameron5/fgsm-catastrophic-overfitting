# Quick Start Guide

This guide will help you get started with the Catastrophic Overfitting demonstration in under 5 minutes.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended, but CPU works too)
- 10 GB free disk space (for CIFAR-10 dataset and checkpoints)

## Installation

### Option 1: Automatic Dependency Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/catastrophic-overfitting-demo.git
cd catastrophic-overfitting-demo

# Check and install dependencies automatically
python check_install.py
```

### Option 2: Manual Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/catastrophic-overfitting-demo.git
cd catastrophic-overfitting-demo

# Install dependencies
pip install -r requirements.txt
```

## Running the Experiments

### Step 1: Demonstrate Catastrophic Overfitting (Standard FGSM)

```bash
python experiments/train_fgsm.py
```

**Expected output:**
- Training will run for 30 epochs (~45 minutes on GPU, ~3 hours on CPU)
- You'll see CO detection around epoch 8-12
- Final PGD-50 accuracy: ~0-5%
- Results saved to `results/catastrophic_overfitting_results.json`
- Visualization saved to `results/co_analysis.png`

### Step 2: Run ATSS Prevention Method

```bash
python experiments/train_atss.py
```

**Expected output:**
- Training will run for 30 epochs
- No CO detection - stable robustness throughout
- Final PGD-50 accuracy: ~45-55%
- Results saved to `results/atss_results.json`
- Visualization saved to `results/atss_analysis.png`

### Step 3: Generate Comparison Visualizations

```bash
python experiments/compare.py
```

**Expected output:**
- Loads results from both experiments
- Creates comprehensive comparison plots
- Prints numerical summary
- Saves visualizations to `results/` directory

## Understanding the Results

### Key Metrics to Watch

1. **PGD-50 Accuracy** (Test Set)
   - Most important metric for robustness
   - Standard FGSM: Drops to ~0%
   - ATSS: Maintains ~45-55%

2. **Robustness Gap** (FGSM - PGD)
   - Standard FGSM: >90% (indicates CO)
   - ATSS: <20% (healthy)

3. **Participation Ratio (PR)**
   - Measures gradient concentration
   - Sharp drop = CO warning sign
   - ATSS maintains stable PR

### What is Catastrophic Overfitting?

When training with standard FGSM:
- ✅ Model achieves high accuracy on FGSM attacks (~100%)
- ❌ Model has zero robustness to PGD attacks (~0%)
- ⚠️ This happens suddenly around epochs 8-12

ATSS prevents this by adapting the step size based on gradient-noise similarity.

## Troubleshooting

### Out of Memory Error

Reduce batch size in `src/config.py`:
```python
BATCH_SIZE = 64  # Default is 128
```

### Slow Training

Expected training times:
- GPU (RTX 3090): ~30-45 minutes per experiment
- GPU (GTX 1080): ~60-90 minutes per experiment
- CPU: ~3-4 hours per experiment

### Import Errors

Make sure you're running from the project root:
```bash
cd catastrophic-overfitting-demo
python experiments/train_fgsm.py
```

## Next Steps

1. Modify hyperparameters in `src/config.py`
2. Try different epsilon values
3. Experiment with other architectures
4. Read the papers cited in README.md

## Getting Help

- Open an issue on GitHub
- Check the full README.md for detailed documentation
- Review the code comments in `src/` files

## Citation

If you use this code in your research, please cite the original papers:
- Wong et al. (2020) - Fast adversarial training
- Mehouachi et al. (2025) - Participation Ratio analysis
- ATSS paper (2024) - Adaptive step size method
