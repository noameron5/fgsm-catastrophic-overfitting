# Catastrophic Overfitting in Fast Adversarial Training

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive implementation demonstrating **Catastrophic Overfitting (CO)** in single-step adversarial training and its prevention using **Adaptive Similarity Step Size (ATSS)** on CIFAR-10.

## 📋 Overview

This repository implements and compares two adversarial training methods:

1. **Standard FGSM Training** - Demonstrates catastrophic overfitting where models maintain high FGSM accuracy (~100%) but PGD robustness collapses to near 0%
2. **ATSS Training** - Prevents catastrophic overfitting through adaptive step sizes based on noise-gradient similarity

### Key Features

- ✅ Complete implementation of catastrophic overfitting phenomenon
- ✅ ATSS prevention method with adaptive step sizing
- ✅ Participation Ratio (PR) analysis for gradient concentration monitoring
- ✅ Comprehensive visualization and comparison tools
- ✅ Multi-epsilon decision boundary analysis
- ✅ Detailed metrics tracking and reporting

## 🔬 Research Background

**Based on:**
- Wong et al. (2020) - "Fast is better than free: Revisiting adversarial training"
- Mehouachi et al. (2025) - "A Noiseless lp Norm Solution for Fast Adversarial Training"
- ATSS paper (2024) - "Avoiding catastrophic overfitting in fast adversarial training"

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/catastrophic-overfitting-demo.git
cd catastrophic-overfitting-demo

# Install dependencies
pip install -r requirements.txt
```

### Run Experiments

```bash
# 1. Demonstrate catastrophic overfitting (Standard FGSM)
python experiments/train_fgsm.py

# 2. Run ATSS prevention method
python experiments/train_atss.py

# 3. Generate comparison visualizations
python experiments/compare.py
```

## 📊 Expected Results

### Catastrophic Overfitting (Standard FGSM)
- **FGSM Test Accuracy**: ~95-100%
- **PGD-50 Test Accuracy**: ~0-5%
- **CO Detection**: Around epoch 10-15
- **Participation Ratio**: Sharp drop at CO onset

### ATSS Prevention
- **FGSM Test Accuracy**: ~85-90%
- **PGD-50 Test Accuracy**: ~45-55%
- **Stability**: Maintained throughout training
- **Participation Ratio**: Stable, no sharp drops

## 📁 Project Structure

```
catastrophic-overfitting-demo/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── setup.py                   # Package installation
├── .gitignore                # Git ignore rules
├── src/                      # Source code
│   ├── __init__.py
│   ├── models.py             # PreActResNet18 architecture
│   ├── attacks.py            # FGSM, PGD, ATSS implementations
│   ├── utils.py              # Training, evaluation, data loading
│   └── config.py             # Configuration constants
├── experiments/              # Experiment scripts
│   ├── __init__.py
│   ├── train_fgsm.py         # CO demonstration
│   ├── train_atss.py         # ATSS prevention
│   └── compare.py            # Comparison visualizations
└── results/                  # Generated results (created at runtime)
    ├── catastrophic_overfitting_results.json
    ├── atss_results.json
    ├── *.png                 # Visualization plots
    └── *.pth                 # Model checkpoints
```

## 🔧 Configuration

Key hyperparameters can be modified in `src/config.py`:

```python
EPSILON = 10/255              # Perturbation budget
LEARNING_RATE = 0.1          # Initial learning rate
EPOCHS = 30                   # Training epochs
BATCH_SIZE = 128             # Batch size
```

## 📈 Metrics Explained

### Participation Ratio (PR1)
- Measures gradient concentration: `PR = (||∇||₁ / ||∇||₂)²`
- **Lower PR** → Gradients concentrated in few dimensions
- **Higher PR** → Gradients spread across many dimensions
- Sharp drops correlate with CO onset

### Robustness Gap
- Difference between FGSM and PGD-50 accuracy
- **Gap > 90%** → Strong indicator of catastrophic overfitting
- **Gap < 10%** → Healthy robust training

## 🎓 Understanding the Results

The key signature of catastrophic overfitting:

1. **High FGSM accuracy** on training set (~100%)
2. **Near-zero PGD accuracy** on test set (~0%)
3. **Sharp PR drop** indicating gradient concentration
4. **Decision boundary distortion** - vulnerability to multi-step attacks

ATSS prevents this by:
- Adapting step size based on noise-gradient similarity
- Maintaining diverse gradient distributions
- Ensuring stable multi-step robustness

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{wong2020fast,
  title={Fast is better than free: Revisiting adversarial training},
  author={Wong, Eric and Rice, Leslie and Kolter, J Zico},
  journal={arXiv preprint arXiv:2001.03994},
  year={2020}
}

@article{mehouachi2025noiseless,
  title={A Noiseless lp Norm Solution for Fast Adversarial Training},
  author={Mehouachi, et al.},
  year={2025}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- CIFAR-10 dataset creators
- Original authors of the adversarial training papers

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact noammeron@mail.tau.ac.il

---

**Note**: This implementation is for educational and research purposes. Training times may vary based on hardware (GPU recommended).
