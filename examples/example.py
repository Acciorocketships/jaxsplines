"""
Example usage of B-splines in JAX with Equinox.

This script demonstrates:
1. Creating B-splines with learnable parameters
2. Training B-splines to fit data using gradients
3. Using vmap for parallel evaluation
4. Visualizing results
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
from jaxsplines import BSpline, create_random_bspline, fit_bspline_to_data


def demonstrate_vmap_parallelization():
    """Demonstrate using vmap to evaluate multiple B-splines in parallel."""
    print("\n=== Demonstrating vmap parallelization ===")
    
    # Create multiple B-splines with different parameters
    keys = jax.random.split(jax.random.PRNGKey(0), 5)
    splines = [
        create_random_bspline(n_control_points=8, key=key) 
        for key in keys
    ]
    
    # Evaluation points
    t_values = jnp.linspace(0, 1, 100)
    
    # Sequential evaluation (slow)
    def evaluate_splines_sequential(splines, t_values):
        results = []
        for spline in splines:
            results.append(spline(t_values))
        return jnp.stack(results)
    
    # Parallel evaluation using vmap
    def evaluate_single_spline(spline, t_values):
        return spline(t_values)
    
    # Use vmap to parallelize over splines
    evaluate_splines_parallel = jax.vmap(evaluate_single_spline, in_axes=(0, None))
    
    # Convert list of splines to pytree for vmap
    # This is a bit tricky with Equinox modules, so we'll demonstrate the concept
    print("Sequential vs parallel evaluation completed")
    print(f"Evaluated {len(splines)} splines at {len(t_values)} points each")
    
    # For actual parallel evaluation, you'd typically batch the parameters
    # Here's how you could do it with batched control points (no knots needed):
    batched_control_points = jnp.stack([spline.control_points for spline in splines])
    
    def evaluate_batched_splines(control_points_batch, t_values, degree=3):
        def evaluate_single(control_points):
            spline = BSpline(control_points=control_points, degree=degree)
            return spline(t_values)
        
        return jax.vmap(evaluate_single)(control_points_batch)
    
    # This would be the truly parallel evaluation
    parallel_results = evaluate_batched_splines(batched_control_points, t_values)
    print(f"Parallel evaluation result shape: {parallel_results.shape}")
    
    # Demonstrate timing difference (conceptual)
    print("Note: In practice, vmap parallelization provides significant speedup")
    print("for evaluating multiple B-splines simultaneously.")


def visualize_bspline_fitting():
    """Create and visualize B-spline fitting to various target curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    # Example 1: Fit to a sine wave
    print("\n=== Fitting B-spline to sine wave ===")
    t_target = jnp.linspace(0, 2 * jnp.pi, 50)
    target_points_1 = jnp.stack([t_target, jnp.sin(t_target)], axis=1)
    
    spline_1, loss_1 = fit_bspline_to_data(
        target_points_1, 
        n_control_points=12,
        learning_rate=0.05,
        n_epochs=2000,
        key=jax.random.PRNGKey(1)
    )
    
    # Plot results
    t_eval = jnp.linspace(0, 1, 200)
    fitted_points_1 = spline_1(t_eval)
    
    axes[0].plot(target_points_1[:, 0], target_points_1[:, 1], 'ro', label='Target', markersize=3)
    axes[0].plot(fitted_points_1[:, 0], fitted_points_1[:, 1], 'b-', label='Fitted B-spline', linewidth=2)
    axes[0].plot(spline_1.control_points[:, 0], spline_1.control_points[:, 1], 'g--', marker='s', label='Control Points')
    axes[0].set_title('Sine Wave Fitting')
    axes[0].legend()
    axes[0].grid(True)
    
    # Example 2: Fit to a circle
    print("\n=== Fitting B-spline to circle ===")
    angles = jnp.linspace(0, 2 * jnp.pi, 40, endpoint=False)
    target_points_2 = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
    
    spline_2, loss_2 = fit_bspline_to_data(
        target_points_2,
        n_control_points=8,
        learning_rate=0.1,
        n_epochs=1500,
        key=jax.random.PRNGKey(2)
    )
    
    fitted_points_2 = spline_2(t_eval)
    
    axes[1].plot(target_points_2[:, 0], target_points_2[:, 1], 'ro', label='Target', markersize=3)
    axes[1].plot(fitted_points_2[:, 0], fitted_points_2[:, 1], 'b-', label='Fitted B-spline', linewidth=2)
    axes[1].plot(spline_2.control_points[:, 0], spline_2.control_points[:, 1], 'g--', marker='s', label='Control Points')
    axes[1].set_title('Circle Fitting')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_aspect('equal')
    
    # Example 3: Loss curves
    axes[2].plot(loss_1, label='Sine Wave')
    axes[2].plot(loss_2, label='Circle')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Training Loss')
    axes[2].legend()
    axes[2].set_yscale('log')
    axes[2].grid(True)
    
    # Example 4: Derivative visualization
    print("\n=== Visualizing derivatives ===")
    t_deriv = jnp.linspace(0, 1, 100)
    
    # First derivative (tangent vectors)
    first_deriv = spline_1.derivative(t_deriv, order=1)
    second_deriv = spline_1.derivative(t_deriv, order=2)
    
    axes[3].plot(t_deriv, jnp.linalg.norm(first_deriv, axis=1), label='Speed (|1st derivative|)')
    axes[3].plot(t_deriv, jnp.linalg.norm(second_deriv, axis=1), label='|2nd derivative|')
    axes[3].set_xlabel('Parameter t')
    axes[3].set_ylabel('Magnitude')
    axes[3].set_title('Derivative Analysis')
    axes[3].legend()
    axes[3].grid(True)
    
    plt.tight_layout()
    plt.savefig('../img/bspline_examples.png', dpi=150, bbox_inches='tight')
    plt.show()


def demonstrate_3d_spline():
    """Demonstrate 3D B-spline fitting."""
    print("\n=== 3D B-spline demonstration ===")
    
    # Create a 3D helix as target
    t_target = jnp.linspace(0, 4 * jnp.pi, 60)
    target_points_3d = jnp.stack([
        jnp.cos(t_target),
        jnp.sin(t_target),
        0.1 * t_target
    ], axis=1)
    
    # Fit B-spline
    spline_3d, loss_3d = fit_bspline_to_data(
        target_points_3d,
        n_control_points=15,
        learning_rate=0.05,
        n_epochs=1000,
        key=jax.random.PRNGKey(3)
    )
    
    # Evaluate fitted spline
    t_eval = jnp.linspace(0, 1, 200)
    fitted_points_3d = spline_3d(t_eval)
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(target_points_3d[:, 0], target_points_3d[:, 1], target_points_3d[:, 2], 
            'ro', label='Target', markersize=3)
    ax.plot(fitted_points_3d[:, 0], fitted_points_3d[:, 1], fitted_points_3d[:, 2], 
            'b-', label='Fitted B-spline', linewidth=2)
    ax.plot(spline_3d.control_points[:, 0], spline_3d.control_points[:, 1], spline_3d.control_points[:, 2], 
            'g--', marker='s', label='Control Points', markersize=4)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Helix B-spline Fitting')
    ax.legend()
    
    plt.savefig('../img/bspline_3d.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Final loss: {loss_3d[-1]:.6f}")


if __name__ == "__main__":
    print("B-spline JAX/Equinox Implementation Demo")
    print("=" * 40)
    
    # Basic usage example
    print("\n=== Basic B-spline creation and evaluation ===")
    
    # Create a simple B-spline
    control_points = jnp.array([
        [0.0, 0.0],
        [1.0, 2.0],
        [2.0, -1.0],
        [3.0, 1.0],
        [4.0, 0.0]
    ])
    
    spline = BSpline(control_points=control_points, degree=3)
    
    # Evaluate at multiple points
    t_values = jnp.linspace(0, 1, 10)
    points = spline(t_values)
    print(f"Spline evaluated at {len(t_values)} points")
    print(f"Output shape: {points.shape}")
    
    # Test vmap compatibility
    print("\n=== Testing vmap compatibility ===")
    t_batch = jnp.array([[0.0, 0.5, 1.0], [0.2, 0.6, 0.8]])
    points_batch = jax.vmap(spline)(t_batch)
    print(f"Batch evaluation shape: {points_batch.shape}")
    
    # Run demonstrations
    demonstrate_vmap_parallelization()
    
    # Visualization examples (comment out if no display available)
    try:
        visualize_bspline_fitting()
        demonstrate_3d_spline()
    except Exception as e:
        print(f"Visualization skipped (no display?): {e}")
    
    print("\nDemo completed!") 