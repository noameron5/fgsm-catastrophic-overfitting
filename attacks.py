"""
Adversarial Attack Implementations.

This module contains implementations of various adversarial attacks:
- FGSM (Fast Gradient Sign Method)
- PGD (Projected Gradient Descent)
- ATSS (Adaptive Similarity Step Size)
"""

import torch
import torch.nn.functional as F


def fgsm_attack(model, X, y, epsilon, clamp=(0, 1), random_start=False):
    """
    Fast Gradient Sign Method (FGSM) attack.
    
    Args:
        model (nn.Module): The model to attack
        X (torch.Tensor): Clean input images
        y (torch.Tensor): True labels
        epsilon (float): Perturbation budget
        clamp (tuple): Min and max values for clamping
        random_start (bool): Whether to use random initialization (RS-FGSM)
    
    Returns:
        torch.Tensor: Adversarial examples
    
    Notes:
        - random_start=False: Standard FGSM (triggers catastrophic overfitting)
        - random_start=True: RS-FGSM (helps delay CO)
    """
    if random_start:
        # Random start FGSM (RS-FGSM) - helps prevent early CO
        delta = torch.zeros_like(X).uniform_(-epsilon, epsilon)
        X_adv = torch.clamp(X + delta, clamp[0], clamp[1])
        X_adv = X_adv.detach().requires_grad_(True)
    else:
        # Standard FGSM (no random start - triggers CO)
        X_adv = X.clone().detach().requires_grad_(True)
    
    # Forward pass
    outputs = model(X_adv)
    loss = F.cross_entropy(outputs, y)
    
    # Backward pass
    loss.backward()
    
    # FGSM step
    grad_sign = X_adv.grad.sign()
    if random_start:
        X_adv = X + epsilon * grad_sign  # Use original X
    else:
        X_adv = X + epsilon * grad_sign
    
    # Clamp to valid range
    X_adv = torch.clamp(X_adv, clamp[0], clamp[1])
    
    return X_adv.detach()


def pgd_attack(model, X, y, epsilon, alpha, num_steps, restarts=1, clamp=(0, 1)):
    """
    Projected Gradient Descent (PGD) attack with multiple restarts.
    
    This is the gold standard for evaluating adversarial robustness.
    
    Args:
        model (nn.Module): The model to attack
        X (torch.Tensor): Clean input images
        y (torch.Tensor): True labels
        epsilon (float): Perturbation budget (L-infinity norm)
        alpha (float): Step size (typically epsilon/4)
        num_steps (int): Number of attack iterations
        restarts (int): Number of random restarts
        clamp (tuple): Min and max values for clamping
    
    Returns:
        torch.Tensor: Adversarial examples with maximum loss
    """
    max_loss = torch.zeros(y.shape[0]).to(X.device)
    max_X_adv = X.clone().detach()
    
    for _ in range(restarts):
        X_adv = X.clone().detach()
        # Random initialization within epsilon ball
        X_adv = X_adv + torch.zeros_like(X_adv).uniform_(-epsilon, epsilon)
        X_adv = torch.clamp(X_adv, clamp[0], clamp[1])
        
        for _ in range(num_steps):
            X_adv.requires_grad_(True)
            outputs = model(X_adv)
            loss = F.cross_entropy(outputs, y, reduction='none')
            loss.sum().backward()
            
            with torch.no_grad():
                # PGD step
                X_adv = X_adv + alpha * X_adv.grad.sign()
                # Project back to epsilon ball
                delta = torch.clamp(X_adv - X, min=-epsilon, max=epsilon)
                X_adv = torch.clamp(X + delta, clamp[0], clamp[1])
        
        # Keep adversarial examples with maximum loss
        with torch.no_grad():
            outputs = model(X_adv)
            loss = F.cross_entropy(outputs, y, reduction='none')
            mask = loss > max_loss
            max_loss[mask] = loss[mask]
            max_X_adv[mask] = X_adv[mask]
    
    return max_X_adv


def atss_attack(model, X, y, epsilon, alpha0, beta, clamp=(0, 1)):
    """
    Adaptive Similarity Step Size (ATSS) attack.
    
    This method prevents catastrophic overfitting by adapting the step size
    based on the similarity between random noise and gradients.
    
    Args:
        model (nn.Module): The model to attack
        X (torch.Tensor): Clean input images
        y (torch.Tensor): True labels
        epsilon (float): Perturbation budget
        alpha0 (float): Base step size (typically equal to epsilon)
        beta (float): Influence coefficient (typically 0.5)
        clamp (tuple): Min and max values for clamping
    
    Returns:
        torch.Tensor: Adversarial examples with adaptive step size
    
    References:
        "Avoiding catastrophic overfitting in fast adversarial training
        with adaptive similarity step size" (2024)
    """
    batch_size = X.size(0)
    
    # Generate random noise
    eta = torch.empty_like(X).uniform_(-1, 1)
    X_noise = torch.clamp(X + epsilon * eta, clamp[0], clamp[1])
    X_noise = X_noise.detach().requires_grad_(True)
    
    # Forward pass
    outputs = model(X_noise)
    loss = F.cross_entropy(outputs, y)
    
    # Backward pass
    loss.backward()
    v = X_noise.grad.detach()
    
    # Flatten for similarity calculations
    eta_flat = eta.view(batch_size, -1)
    v_flat = v.view(batch_size, -1)
    
    # Compute Euclidean distances
    D = torch.sqrt(((eta_flat - v_flat) ** 2).sum(dim=1))
    
    # Compute Cosine similarities
    C = (eta_flat * v_flat).sum(dim=1) / (
        eta_flat.norm(2, dim=1) * v_flat.norm(2, dim=1).clamp(min=1e-9)
    )
    
    # Normalize distances
    mean_D, std_D = D.mean(), D.std().clamp(min=1e-9)
    S_ed = (D - mean_D) / std_D
    
    # Normalize cosine values
    mean_C, std_C = C.mean(), C.std().clamp(min=1e-9)
    S_cos = (C - mean_C) / std_C
    
    # Overall similarity score
    s = S_ed + S_cos
    
    # Adaptive step size
    alpha = (1 - beta * s) * alpha0
    alpha = alpha.view(batch_size, 1, 1, 1)  # Broadcast to image dimensions
    
    # Generate adversarial examples
    grad_sign = v.sign()
    X_adv = X_noise + alpha * grad_sign
    
    # Clamp to epsilon ball around original X
    X_adv = torch.clamp(X_adv, X - epsilon, X + epsilon)
    X_adv = torch.clamp(X_adv, clamp[0], clamp[1])
    
    return X_adv.detach()
