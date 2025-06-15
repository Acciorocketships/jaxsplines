#!/usr/bin/env python3
"""
Test JAX compatibility of the monotonicity projection methods.
"""

import jax
import jax.numpy as jnp
from jaxsplines import BSpline

def test_jax_compatibility():
    """Test that monotonicity projection is JAX-compatible."""
    
    print("=== Testing JAX Compatibility of Monotonicity Projection ===")
    
    # Create a test B-spline
    control_points = jnp.array([
        [3.0, 0.0],  # Non-monotonic in x
        [1.0, 1.0],
        [4.0, 2.0],
        [2.0, 3.0],
        [5.0, 4.0]
    ])
    
    spline = BSpline(control_points=control_points, degree=3)
    
    print(f"Original control points (x-dim): {control_points[:, 0]}")
    
    # Test 1: Basic projection
    projected_spline = spline.project_to_monotonic()
    print(f"Projected control points (x-dim): {projected_spline.control_points[:, 0]}")
    print(f"Projected control points (y-dim): {projected_spline.control_points[:, 1]}")
    
    # Test 2: Simple projection
    simple_projected = spline.project_to_monotonic_simple()
    print(f"Simple projected (x-dim): {simple_projected.control_points[:, 0]}")
    print(f"Simple projected (y-dim): {simple_projected.control_points[:, 1]}")
    
    # Test 3: JAX transformations
    print("\n=== Testing JAX Transformations ===")
    
    # Test JIT compilation
    @jax.jit
    def jit_project(spline):
        return spline.project_to_monotonic()
    
    jit_result = jit_project(spline)
    print(f"JIT compiled result (x-dim): {jit_result.control_points[:, 0]}")
    print(f"JIT compiled result (y-dim): {jit_result.control_points[:, 1]}")
    
    # Test gradient computation
    def loss_with_projection(control_points):
        spline = BSpline(control_points=control_points, degree=3)
        projected = spline.project_to_monotonic()
        # Simple loss: sum of squares of projected control points in both dimensions
        return jnp.sum(projected.control_points ** 2)
    
    print("\n=== Testing Gradient Computation ===")
    try:
        loss_value, grads = jax.value_and_grad(loss_with_projection)(control_points)
        print(f"✅ Gradient computation successful!")
        print(f"   Loss value: {loss_value:.6f}")
        print(f"   Gradient shape: {grads.shape}")
        print(f"   Gradient (x-dim): {grads[:, 0]}")
    except Exception as e:
        print(f"❌ Gradient computation failed: {e}")
    
    # Test simple version
    def loss_with_simple_projection(control_points):
        spline = BSpline(control_points=control_points, degree=3)
        projected = spline.project_to_monotonic_simple()
        return jnp.sum(projected.control_points ** 2)
    
    print("\n=== Testing Simple Version Gradients ===")
    try:
        loss_value, grads = jax.value_and_grad(loss_with_simple_projection)(control_points)
        print(f"✅ Simple gradient computation successful!")
        print(f"   Loss value: {loss_value:.6f}")
        print(f"   Gradient shape: {grads.shape}")
        print(f"   Gradient (x-dim): {grads[:, 0]}")
    except Exception as e:
        print(f"❌ Simple gradient computation failed: {e}")
    
    # Test vmap compatibility
    print("\n=== Testing vmap Compatibility ===")
    
    def project_batch(control_points_batch):
        """Project a batch of splines."""
        def project_single(cp):
            spline = BSpline(control_points=cp, degree=3)
            return spline.project_to_monotonic_simple().control_points
        
        return jax.vmap(project_single)(control_points_batch)
    
    # Create a batch of control points
    batch_control_points = jnp.stack([
        control_points,
        control_points + 1.0,
        control_points - 0.5
    ])
    
    try:
        batch_results = project_batch(batch_control_points)
        print(f"✅ vmap compatibility successful!")
        print(f"   Batch input shape: {batch_control_points.shape}")
        print(f"   Batch output shape: {batch_results.shape}")
        print(f"   First result (x-dim): {batch_results[0, :, 0]}")
    except Exception as e:
        print(f"❌ vmap compatibility failed: {e}")
    
    # Test monotonicity verification
    print("\n=== Verifying Monotonicity ===")
    for i, (name, result) in enumerate([
        ("Original", spline),
        ("Projected", projected_spline),
        ("Simple", simple_projected),
        ("JIT", jit_result)
    ]):
        x_coords = result.control_points[:, 0]
        is_monotonic = jnp.all(x_coords[1:] >= x_coords[:-1])
        print(f"   {name}: {x_coords} -> Monotonic: {is_monotonic}")


if __name__ == "__main__":
    test_jax_compatibility() 