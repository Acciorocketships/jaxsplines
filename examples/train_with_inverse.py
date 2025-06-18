#!/usr/bin/env python3
"""
Train a monotonic B-spline to fit data and visualize both the function and its inverse.

This example demonstrates:
1. Training a B-spline using gradient descent with monotonic projection
2. Static visualization comparing monotonic vs free training
3. How the inverse behaves with monotonic constraints
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jaxsplines import BSpline

# Set random seed for reproducibility
jax.config.update('jax_enable_x64', True)
np.random.seed(42)
key = jax.random.PRNGKey(42)

def target_function(x):
    """Target function to learn - a smooth monotonic function."""
    # Monotonic sigmoid-like function that goes from ~0.1 to ~0.9
    return 0.1 + 0.8 / (1 + jnp.exp(-6 * (x - 0.5)))

def loss_function(control_points_y, x_data, y_data):
    """Loss function for training the B-spline.
    
    Args:
        control_points_y: Shape (n_control,) - just the y-values of control points
        x_data: Shape (n_points,) - x coordinates where we want to evaluate
        y_data: Shape (n_points,) - target y values
    """
    # Create 2D control points with x-coordinates evenly spaced
    n_control = len(control_points_y)
    control_x = jnp.linspace(0, 1, n_control)
    control_points = jnp.column_stack([control_x, control_points_y])
    
    spline = BSpline(control_points=control_points, degree=3)
    
    # Use inversion to map x_data to parameter space (handles arrays automatically)
    t_data = spline.invert(x_data, dimension=0, assume_monotonic=True)

    # Evaluate spline at the inverted parameters and extract y-coordinates
    y_pred = spline(t_data)[:, 1]  # Shape (n_points,) - just y-coordinates
    
    return jnp.mean((y_pred - y_data) ** 2)

def monotonic_training_comparison():
    """Create a comparison showing monotonic vs free training results."""
    print("=== Monotonic vs Free Training Comparison ===")
    
    # Generate training data
    n_points = 50
    x_train = jnp.linspace(0, 1, n_points)
    y_train = target_function(x_train)
    
    # Add some noise
    key_noise = jax.random.PRNGKey(123)
    noise = 0.02 * jax.random.normal(key_noise, shape=y_train.shape)
    y_train_noisy = y_train + noise
    
    print(f"Target function range: {float(jnp.min(y_train)):.3f} to {float(jnp.max(y_train)):.3f}")
    
    # Initial control points (just y-coordinates)
    n_control = 8
    initial_control_y = jnp.linspace(0.2, 0.8, n_control)
    
    print(f"Training data: {n_points} points")
    print(f"Control points: {n_control}")
    print(f"Initial control points y-values: {initial_control_y}")
    
    # Train WITH monotonic projection
    control_y_mono = initial_control_y.copy()
    learning_rate = 0.02
    epochs = 500
    loss_fn = jax.jit(loss_function)
    grad_fn = jax.jit(jax.grad(loss_function))
    
    print("Training with monotonic projection...")
    losses_mono = []
    for epoch in range(epochs):
        grad = grad_fn(control_y_mono, x_train, y_train_noisy)
        control_y_mono = control_y_mono - learning_rate * grad
        
        # Apply monotonic projection
        control_x = jnp.linspace(0, 1, n_control)
        full_control_points = jnp.column_stack([control_x, control_y_mono])
        spline = BSpline(control_points=full_control_points, degree=3)
        projected = spline.project_to_monotonic(method="exact")
        control_y_mono = projected.control_points[:, 1]  # Extract y-coordinates
        
        current_loss = loss_fn(control_y_mono, x_train, y_train_noisy)
        losses_mono.append(float(current_loss))
        
        if epoch % 100 == 0:
            print(f"  Epoch {epoch}: Loss = {current_loss:.6f}")
    
    # Train WITHOUT monotonic projection (for comparison)
    print("Training without monotonic projection...")
    control_y_free = initial_control_y.copy()
    losses_free = []
    for epoch in range(epochs):
        grad = grad_fn(control_y_free, x_train, y_train_noisy)
        control_y_free = control_y_free - learning_rate * grad
        
        current_loss = loss_fn(control_y_free, x_train, y_train_noisy)
        losses_free.append(float(current_loss))
        
        if epoch % 100 == 0:
            print(f"  Epoch {epoch}: Loss = {current_loss:.6f}")
    
    # Create comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Monotonic vs Free Training Comparison', fontsize=16)
    
    t_eval = jnp.linspace(0, 1, 200)
    x_eval = jnp.linspace(0, 1, 200)
    control_x = jnp.linspace(0, 1, n_control)
    
    # Monotonic training result
    mono_control_points = jnp.column_stack([control_x, control_y_mono])
    mono_spline = BSpline(control_points=mono_control_points, degree=3)
    mono_eval = mono_spline(t_eval)
    
    ax1.set_title('With Monotonic Projection')
    ax1.plot(x_eval, target_function(x_eval), 'b-', label='Target', linewidth=2)
    ax1.scatter(x_train, y_train_noisy, c='red', s=20, alpha=0.6, label='Noisy Data')
    ax1.plot(mono_eval[:, 0], mono_eval[:, 1], 'g-', label='Monotonic Fit', linewidth=2)
    ax1.scatter(mono_control_points[:, 0], mono_control_points[:, 1], c='orange', s=50, marker='s')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Free training result
    free_control_points = jnp.column_stack([control_x, control_y_free])
    free_spline = BSpline(control_points=free_control_points, degree=3)
    free_eval = free_spline(t_eval)
    
    ax2.set_title('Without Monotonic Projection')
    ax2.plot(x_eval, target_function(x_eval), 'b-', label='Target', linewidth=2)
    ax2.scatter(x_train, y_train_noisy, c='red', s=20, alpha=0.6, label='Noisy Data')
    ax2.plot(free_eval[:, 0], free_eval[:, 1], 'purple', label='Free Fit', linewidth=2)
    ax2.scatter(free_control_points[:, 0], free_control_points[:, 1], c='orange', s=50, marker='s')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # Training losses comparison
    ax3.set_title('Training Loss Comparison')
    ax3.plot(losses_mono, 'g-', label='With Monotonic Projection', linewidth=2)
    ax3.plot(losses_free, 'purple', label='Without Projection', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('MSE Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Inverse comparison
    ax4.set_title('Inverse Functions')
    ax4.plot(mono_eval[:, 1], mono_eval[:, 0], 'g-', label='Monotonic Inverse', linewidth=2)
    
    # Test if free spline is monotonic for inverse
    try:
        is_free_monotonic = free_spline.check_monotonic(1)
        if is_free_monotonic:
            ax4.plot(free_eval[:, 1], free_eval[:, 0], 'purple', label='Free Inverse', linewidth=2)
        else:
            ax4.text(0.5, 0.2, 'Free spline not monotonic\n(no unique inverse)', 
                    ha='center', va='center', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    except:
        pass
    
    # Test analytical inversion for monotonic spline
    try:
        y_test = jnp.linspace(0.2, 0.8, 10)
        x_inverted = []
        for y_val in y_test:
            try:
                t_inv = mono_spline.invert(float(y_val), dimension=1, assume_monotonic=True)
                x_inv = mono_spline(t_inv)[0]
                x_inverted.append(float(x_inv))
            except:
                x_inverted.append(np.nan)
        
        valid_mask = ~np.isnan(x_inverted)
        if np.any(valid_mask):
            ax4.scatter(y_test[valid_mask], np.array(x_inverted)[valid_mask], 
                      c='red', s=40, marker='o', label='Analytical Inverse')
    except:
        pass
    
    ax4.set_xlabel('y')
    ax4.set_ylabel('x')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    # Print final comparison
    print(f"\n=== Final Results ===")
    print(f"Final loss with monotonic projection: {losses_mono[-1]:.6f}")
    print(f"Final loss without projection: {losses_free[-1]:.6f}")
    print(f"Monotonic spline is monotonic: {mono_spline.check_monotonic(1)}")
    print(f"Free spline is monotonic: {free_spline.check_monotonic(1)}")
    
    # Test inverse functionality extensively
    print("\nTesting inverse functionality...")
    try:
        # Test a range of inverse computations
        test_y_values = jnp.linspace(0.15, 0.85, 8)
        print("Inverse tests:")
        for y_val in test_y_values:
            t_inv = mono_spline.invert(float(y_val), dimension=1, assume_monotonic=True)
            x_inv = mono_spline(t_inv)[0]
            # Verify round-trip
            y_verify = mono_spline(mono_spline.invert(float(y_val), dimension=1, assume_monotonic=True))[1]
            error = abs(float(y_verify) - float(y_val))
            print(f"  y={y_val:.3f} -> x={x_inv:.6f} (t={t_inv:.6f}), round-trip error: {error:.2e}")
    except Exception as e:
        print(f"  Inverse computation failed: {e}")
    
    return mono_spline, free_spline

if __name__ == "__main__":
    monotonic_training_comparison()
    print("\n✅ Monotonic training comparison completed!") 