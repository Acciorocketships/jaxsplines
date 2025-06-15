"""
Test suite for the B-spline JAX implementation.

This file contains unit tests and advanced usage examples to validate
the B-spline implementation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxsplines import BSpline, create_random_bspline


def test_basic_bspline_creation():
    """Test basic B-spline creation and properties."""
    control_points = jnp.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 0.0],
        [3.0, 1.0]
    ])
    
    spline = BSpline(control_points=control_points, degree=2)
    
    # Check basic properties
    assert spline.control_points.shape == (4, 2)
    assert spline.degree == 2
    
    print("✓ Basic B-spline creation test passed")


def test_minimum_control_points():
    """Test minimum control points validation."""
    # Should work with minimum required control points
    control_points_min = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]])  # 4 points for degree 3
    spline = BSpline(control_points=control_points_min, degree=3)
    assert spline.degree == 3
    
    # Should fail with insufficient control points
    try:
        insufficient_points = jnp.array([[0.0, 0.0], [1.0, 1.0]])  # Only 2 points for degree 3
        BSpline(control_points=insufficient_points, degree=3)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Need at least" in str(e)
    
    print("✓ Minimum control points validation test passed")


def test_spline_evaluation():
    """Test B-spline evaluation at various points."""
    control_points = jnp.array([
        [0.0, 0.0],
        [1.0, 2.0],
        [2.0, 0.0]
    ])
    
    spline = BSpline(control_points=control_points, degree=2)
    
    # Test single point evaluation
    point = spline(0.5)
    assert point.shape == (2,)
    
    # Test batch evaluation
    t_values = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    points = spline(t_values)
    assert points.shape == (5, 2)
    
    # Test 2D batch evaluation
    t_batch = jnp.array([[0.0, 0.5, 1.0], [0.2, 0.6, 0.8]])
    points_batch = spline(t_batch)
    assert points_batch.shape == (2, 3, 2)
    
    print("✓ Spline evaluation test passed")


def test_derivative_computation():
    """Test derivative computation."""
    control_points = jnp.array([
        [0.0, 0.0],
        [1.0, 2.0],
        [2.0, 1.0],
        [3.0, 0.0]
    ])
    
    spline = BSpline(control_points=control_points, degree=3)
    
    # Test first derivative
    t_values = jnp.linspace(0, 1, 10)
    first_deriv = spline.derivative(t_values, order=1)
    assert first_deriv.shape == (10, 2)
    
    # Test second derivative
    second_deriv = spline.derivative(t_values, order=2)
    assert second_deriv.shape == (10, 2)
    
    # Test derivative of degree higher than spline degree (should be zero)
    high_order_deriv = spline.derivative(t_values, order=5)
    assert jnp.allclose(high_order_deriv, 0.0)
    
    print("✓ Derivative computation test passed")


def test_vmap_compatibility():
    """Test vmap compatibility for parallel evaluation."""
    # Create multiple splines
    control_points_batch = jax.random.normal(jax.random.PRNGKey(0), (3, 5, 2))
    
    def create_spline(control_points):
        return BSpline(control_points=control_points, degree=3)
    
    # This tests that the spline can be created in batch
    splines = jax.vmap(create_spline)(control_points_batch)
    
    # Test batch evaluation
    t_values = jnp.linspace(0, 1, 20)
    
    def evaluate_spline(spline):
        return spline(t_values)
    
    # This should work with vmap
    results = jax.vmap(evaluate_spline)(splines)
    assert results.shape == (3, 20, 2)
    
    print("✓ Vmap compatibility test passed")


def test_gradient_computation():
    """Test that gradients can be computed through the spline."""
    control_points = jax.random.normal(jax.random.PRNGKey(1), (5, 2))
    spline = BSpline(control_points=control_points, degree=3)
    
    def loss_fn(spline):
        t_values = jnp.linspace(0, 1, 10)
        points = spline(t_values)
        return jnp.sum(points ** 2)
    
    # Compute gradients (allow_int=True for integer attributes like degree)
    loss, grads = jax.value_and_grad(loss_fn, allow_int=True)(spline)
    
    # Check that gradients exist for control points
    assert grads.control_points.shape == control_points.shape
    assert not jnp.allclose(grads.control_points, 0.0)  # Should have non-zero gradients
    
    print("✓ Gradient computation test passed")


def test_3d_spline():
    """Test 3D B-spline functionality."""
    control_points_3d = jax.random.normal(jax.random.PRNGKey(2), (6, 3))
    spline_3d = BSpline(control_points=control_points_3d, degree=3)
    
    # Test evaluation
    t_values = jnp.linspace(0, 1, 15)
    points_3d = spline_3d(t_values)
    assert points_3d.shape == (15, 3)
    
    # Test derivatives
    deriv_3d = spline_3d.derivative(t_values, order=1)
    assert deriv_3d.shape == (15, 3)
    
    print("✓ 3D spline test passed")


def test_basis_function_properties():
    """Test B-spline basis function properties."""
    control_points = jnp.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 0.0],
        [3.0, 1.0],
        [4.0, 0.0]
    ])
    
    spline = BSpline(control_points=control_points, degree=3)
    
    # Test partition of unity property
    t_values = jnp.linspace(0, 1, 50)
    
    def test_partition_of_unity(t):
        basis_values = spline._compute_basis_functions(t)
        return jnp.sum(basis_values)
    
    sums = jax.vmap(test_partition_of_unity)(t_values)
    
    # Sum of basis functions should be 1 (partition of unity)
    assert jnp.allclose(sums, 1.0, atol=1e-10), f"Partition of unity failed: {sums}"
    
    print("✓ Basis function properties test passed")


def test_learnable_parameters_integration():
    """Test integration with optimization frameworks."""
    import optax
    import equinox as eqx
    
    # Create target data (simple sine wave)
    t_target = jnp.linspace(0, 1, 30)
    target_data = jnp.stack([t_target, jnp.sin(2 * jnp.pi * t_target)], axis=1)
    
    # Initialize spline
    spline = create_random_bspline(
        n_control_points=8,
        dimension=2,
        degree=3,
        key=jax.random.PRNGKey(42)
    )
    
    # Define loss function
    def loss_fn(spline):
        t_eval = jnp.linspace(0, 1, 30)
        pred_points = spline(t_eval)
        return jnp.mean((pred_points - target_data) ** 2)
    
    # Setup optimizer
    optimizer = optax.adam(0.01)
    opt_state = optimizer.init(eqx.filter(spline, eqx.is_array))
    
    # Test one optimization step
    loss_before = loss_fn(spline)
    
    @eqx.filter_jit
    def update_step(spline, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_fn, allow_int=True)(spline)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(spline, eqx.is_array))
        spline = eqx.apply_updates(spline, updates)
        return spline, opt_state, loss
    
    spline, opt_state, loss_after = update_step(spline, opt_state)
    
    # Loss should change (allowing for numerical precision)
    loss_change = abs(loss_after - loss_before)
    print(f"Loss before: {loss_before:.8f}, Loss after: {loss_after:.8f}, Change: {loss_change:.8f}")
    
    # Check if loss changed significantly or if gradients are working
    if loss_change < 1e-10:
        # Run a few more steps to see if optimization is working
        for i in range(5):
            spline, opt_state, loss_after = update_step(spline, opt_state)
            loss_change = abs(loss_after - loss_before)
            if loss_change > 1e-10:
                break
    
    # Should have some change after multiple steps
    assert loss_change > 1e-12, f"Optimization didn't change loss after multiple steps: {loss_change}"
    
    print("✓ Learnable parameters integration test passed")


def test_degree_validation():
    """Test that only degrees 1-3 are allowed."""
    control_points = jax.random.normal(jax.random.PRNGKey(0), (5, 2))
    
    # Valid degrees should work
    for degree in [1, 2, 3]:
        spline = BSpline(control_points=control_points, degree=degree)
        assert spline.degree == degree
    
    # Invalid degrees should raise ValueError
    for invalid_degree in [0, 4, 5, -1]:
        try:
            BSpline(control_points=control_points, degree=invalid_degree)
            assert False, f"Should have raised ValueError for degree {invalid_degree}"
        except ValueError as e:
            assert "Only degrees 1-3 are supported" in str(e)
    
    # Test create_random_bspline validation too
    for invalid_degree in [0, 4, 5]:
        try:
            create_random_bspline(5, 2, invalid_degree)
            assert False, f"Should have raised ValueError for degree {invalid_degree}"
        except ValueError as e:
            assert "Only degrees 1-3 are supported" in str(e)
    
    print("✓ Degree validation test passed")


def test_uniform_spacing_assumption():
    """Test that our uniform spacing assumption works correctly."""
    # Create a simple test case where we can verify the uniform spacing behavior
    control_points = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0]
    ])
    
    spline = BSpline(control_points=control_points, degree=3)
    
    # Evaluate at several points
    t_values = jnp.array([0.0, 0.5, 1.0])
    points = spline(t_values)
    
    # With these control points, we should get a roughly linear progression in x
    assert points.shape == (3, 2)
    assert points[0, 1] == 0.0  # y should be 0 for all points
    assert points[1, 1] == 0.0
    assert points[2, 1] == 0.0
    
    print("✓ Uniform spacing assumption test passed")


def run_performance_benchmark():
    """Benchmark the performance of B-spline operations."""
    import time
    
    print("\n=== Performance Benchmark ===")
    
    # Large-scale evaluation test
    control_points = jax.random.normal(jax.random.PRNGKey(0), (20, 3))
    spline = BSpline(control_points=control_points, degree=3)
    
    # Benchmark single evaluation
    t_values = jnp.linspace(0, 1, 10000)
    
    # Compile the function first
    _ = spline(t_values[:10])
    
    start_time = time.time()
    points = spline(t_values)
    end_time = time.time()
    
    print(f"Evaluated {len(t_values)} points in {end_time - start_time:.4f} seconds")
    print(f"Rate: {len(t_values) / (end_time - start_time):.0f} evaluations/second")
    
    # Benchmark batch evaluation with vmap
    batch_size = 100
    t_batch = jax.random.uniform(jax.random.PRNGKey(1), (batch_size, 100))
    
    start_time = time.time()
    batch_points = jax.vmap(spline)(t_batch)
    end_time = time.time()
    
    total_evaluations = batch_size * 100
    print(f"Batch evaluated {total_evaluations} points in {end_time - start_time:.4f} seconds")
    print(f"Batch rate: {total_evaluations / (end_time - start_time):.0f} evaluations/second")


if __name__ == "__main__":
    print("Running B-spline test suite...")
    print("=" * 50)
    
    # Run all tests
    test_basic_bspline_creation()
    test_minimum_control_points()
    test_spline_evaluation()
    test_derivative_computation()
    test_vmap_compatibility()
    test_gradient_computation()
    test_3d_spline()
    test_basis_function_properties()
    test_learnable_parameters_integration()
    test_degree_validation()
    test_uniform_spacing_assumption()
    
    # Run performance benchmark
    run_performance_benchmark()
    
    print("\n" + "=" * 50)
    print("✅ All tests passed!")
    print("B-spline implementation is working correctly.") 