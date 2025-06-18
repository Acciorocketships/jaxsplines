#!/usr/bin/env python3
"""
Comprehensive tests for monotonicity checking, projection, and JAX compatibility.
Combines debugging, mathematical validation, and JAX integration tests.
"""

import jax
import jax.numpy as jnp
from jaxsplines import BSpline
import numpy as np

def test_monotonic_checking():
    """Test monotonicity checking with various cases."""
    print("=== Testing Monotonicity Checking ===")
    
    # Test Case 1: Perfectly linear (should definitely be monotonic)
    print("\n1. Perfectly Linear Case:")
    linear_points = jnp.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0]
    ])
    
    linear_spline = BSpline(control_points=linear_points, degree=3)
    is_mono_x = linear_spline.check_monotonic(0)
    is_mono_y = linear_spline.check_monotonic(1)
    
    print(f"  Control points: {linear_points}")
    print(f"  Monotonic in x: {is_mono_x}")
    print(f"  Monotonic in y: {is_mono_y}")
    
    # Verify with sampling
    t_vals = jnp.linspace(0, 1, 21)
    spline_vals = linear_spline(t_vals)
    x_diff = jnp.diff(spline_vals[:, 0])
    y_diff = jnp.diff(spline_vals[:, 1])
    
    print(f"  Sampled x differences: min={jnp.min(x_diff):.6f}, max={jnp.max(x_diff):.6f}")
    print(f"  Sampled y differences: min={jnp.min(y_diff):.6f}, max={jnp.max(y_diff):.6f}")
    
    # Test Case 2: Curved but monotonic
    print("\n2. Curved Monotonic Case:")
    curved_points = jnp.array([
        [0.0, 0.0],
        [0.2, 0.3],  # Slightly above linear
        [0.8, 0.7],  # Slightly below linear but still increasing
        [1.0, 1.0]
    ])
    
    curved_spline = BSpline(control_points=curved_points, degree=3)
    is_mono_x = curved_spline.check_monotonic(0)
    is_mono_y = curved_spline.check_monotonic(1)
    
    print(f"  Control points: {curved_points}")
    print(f"  Monotonic in x: {is_mono_x}")
    print(f"  Monotonic in y: {is_mono_y}")
    
    # Test Case 3: Non-monotonic case
    print("\n3. Non-Monotonic Case:")
    non_mono_points = jnp.array([
        [0.0, 0.0],
        [2.0, 1.0],  # Goes up in x
        [1.0, 2.0],  # Then down in x - not monotonic
        [3.0, 3.0]
    ])
    
    non_mono_spline = BSpline(control_points=non_mono_points, degree=3)
    is_mono_x = non_mono_spline.check_monotonic(0)
    is_mono_y = non_mono_spline.check_monotonic(1)
    
    print(f"  Control points: {non_mono_points}")
    print(f"  Monotonic in x: {is_mono_x} (should be False)")
    print(f"  Monotonic in y: {is_mono_y} (should be True)")


def test_monotonic_projection():
    """Test monotonic projection methods."""
    print("\n=== Testing Monotonic Projection ===")
    
    # Create a non-monotonic spline
    control_points = jnp.array([
        [3.0, 0.0],  # Non-monotonic in x
        [1.0, 1.0],
        [4.0, 2.0],
        [2.0, 3.0],
        [5.0, 4.0]
    ])
    
    spline = BSpline(control_points=control_points, degree=3)
    
    print(f"Original control points (x-dim): {control_points[:, 0]}")
    print(f"Original is monotonic in x: {spline.check_monotonic(0)}")
    
    # Test exact projection
    exact_projected = spline.project_to_monotonic(method="exact")
    print(f"Exact projected (x-dim): {exact_projected.control_points[:, 0]}")
    print(f"Exact projected is monotonic in x: {exact_projected.check_monotonic(0)}")
    
    # Test simple projection
    simple_projected = spline.project_to_monotonic(method="simple")
    print(f"Simple projected (x-dim): {simple_projected.control_points[:, 0]}")
    print(f"Simple projected is monotonic in x: {simple_projected.check_monotonic(0)}")
    
    # Verify both projections produce monotonic results
    all_monotonic = (
        exact_projected.check_monotonic(0) and exact_projected.check_monotonic(1) and
        simple_projected.check_monotonic(0) and simple_projected.check_monotonic(1)
    )
    print(f"All projections monotonic: {all_monotonic}")


def test_projection_method_options():
    """Test that both simple and exact projection methods work."""
    print("\n=== Testing Projection Method Options ===")
    
    # Create a non-monotonic spline
    control_points = jnp.array([
        [3.0, 0.0],  # Non-monotonic
        [1.0, 1.0],
        [4.0, 2.0],
        [2.0, 3.0]
    ])
    
    spline = BSpline(control_points=control_points, degree=3)
    print(f"Original control points (x-dim): {control_points[:, 0]}")
    print(f"Original is monotonic: {spline.check_monotonic(0)}")
    
    # Test simple method (default)
    simple_projected = spline.project_to_monotonic()  # Should use "simple" by default
    print(f"Simple method (default) result (x-dim): {simple_projected.control_points[:, 0]}")
    print(f"Simple method is monotonic: {simple_projected.check_monotonic(0)}")
    
    # Test simple method (explicit)
    simple_explicit = spline.project_to_monotonic(method="simple")
    print(f"Simple method (explicit) result (x-dim): {simple_explicit.control_points[:, 0]}")
    print(f"Simple explicit is monotonic: {simple_explicit.check_monotonic(0)}")
    
    # Test exact method
    exact_projected = spline.project_to_monotonic(method="exact")
    print(f"Exact method result (x-dim): {exact_projected.control_points[:, 0]}")
    print(f"Exact method is monotonic: {exact_projected.check_monotonic(0)}")
    
    # Test invalid method
    try:
        invalid_projected = spline.project_to_monotonic(method="invalid")
        print(f"❌ Invalid method should have failed")
    except ValueError as e:
        print(f"✅ Invalid method correctly rejected: {e}")
    
    # Verify that both methods work
    both_work = (simple_projected.check_monotonic(0) and 
                simple_projected.check_monotonic(1) and
                exact_projected.check_monotonic(0) and 
                exact_projected.check_monotonic(1))
    print(f"Both methods produce monotonic results: {both_work}")


def test_jax_compatibility():
    """Test JAX compatibility of monotonic operations."""
    print("\n=== Testing JAX Compatibility ===")
    
    control_points = jnp.array([
        [3.0, 0.0],  # Non-monotonic
        [1.0, 1.0],
        [4.0, 2.0],
        [2.0, 3.0]
    ])
    
    spline = BSpline(control_points=control_points, degree=3)
    
    # Test 1: JIT compilation
    @jax.jit
    def jit_project(spline):
        return spline.project_to_monotonic(method="simple")
    
    try:
        jit_result = jit_project(spline)
        print("✅ JIT compilation successful")
        print(f"   JIT result (x-dim): {jit_result.control_points[:, 0]}")
    except Exception as e:
        print(f"❌ JIT compilation failed: {e}")
    
    # Test 2: Gradient computation
    def loss_with_projection(control_points):
        spline = BSpline(control_points=control_points, degree=3)
        projected = spline.project_to_monotonic(method="simple")
        return jnp.sum(projected.control_points ** 2)
    
    try:
        loss_value, grads = jax.value_and_grad(loss_with_projection)(control_points)
        print("✅ Gradient computation successful")
        print(f"   Loss value: {loss_value:.6f}")
        print(f"   Gradient shape: {grads.shape}")
    except Exception as e:
        print(f"❌ Gradient computation failed: {e}")
    
    # Test 3: vmap compatibility
    def project_batch(control_points_batch):
        def project_single(cp):
            spline = BSpline(control_points=cp, degree=3)
            return spline.project_to_monotonic(method="simple").control_points
        
        return jax.vmap(project_single)(control_points_batch)
    
    # Create a batch
    batch_control_points = jnp.stack([
        control_points,
        control_points + 1.0,
        control_points - 0.5
    ])
    
    try:
        batch_results = project_batch(batch_control_points)
        print("✅ vmap compatibility successful")
        print(f"   Batch input shape: {batch_control_points.shape}")
        print(f"   Batch output shape: {batch_results.shape}")
    except Exception as e:
        print(f"❌ vmap compatibility failed: {e}")


def test_mathematical_conditions():
    """Test the mathematical conditions for monotonicity in detail."""
    print("\n=== Testing Mathematical Conditions ===")
    
    # Test with a known problematic case from training
    training_points = jnp.array([
        [0.0, 0.19694649],
        [0.2, 0.29187372],
        [0.4, 0.41482866],
        [0.6, 0.5851713],
        [0.8, 0.70812637],
        [1.0, 0.8030536]
    ])
    
    training_spline = BSpline(control_points=training_points, degree=3)
    is_mono_x = training_spline.check_monotonic(0)
    is_mono_y = training_spline.check_monotonic(1)
    
    print(f"Training case monotonic in x: {is_mono_x}")
    print(f"Training case monotonic in y: {is_mono_y}")
    
    # Sample to verify empirically
    t_vals = jnp.linspace(0, 1, 51)
    spline_vals = training_spline(t_vals)
    x_diff = jnp.diff(spline_vals[:, 0])
    y_diff = jnp.diff(spline_vals[:, 1])
    
    x_monotonic_empirical = jnp.all(x_diff >= -1e-10)  # Allow tiny numerical errors
    y_monotonic_empirical = jnp.all(y_diff >= -1e-10)
    
    print(f"Empirical sampling - x monotonic: {x_monotonic_empirical}")
    print(f"Empirical sampling - y monotonic: {y_monotonic_empirical}")
    print(f"x differences range: [{jnp.min(x_diff):.6f}, {jnp.max(x_diff):.6f}]")
    print(f"y differences range: [{jnp.min(y_diff):.6f}, {jnp.max(y_diff):.6f}]")
    
    # Check mathematical vs empirical consistency
    math_vs_empirical_x = (is_mono_x == x_monotonic_empirical)
    math_vs_empirical_y = (is_mono_y == y_monotonic_empirical)
    
    print(f"Mathematical check matches empirical (x): {math_vs_empirical_x}")
    print(f"Mathematical check matches empirical (y): {math_vs_empirical_y}")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n=== Testing Edge Cases ===")
    
    # Test with different degrees
    for degree in [1, 2, 3]:
        print(f"\nTesting degree {degree}:")
        
        if degree == 1:
            points = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        elif degree == 2:
            points = jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        else:  # degree == 3
            points = jnp.array([[0.0, 0.0], [0.3, 0.3], [0.7, 0.7], [1.0, 1.0]])
        
        spline = BSpline(control_points=points, degree=degree)
        is_mono = spline.check_monotonic(0)
        print(f"   Degree {degree} linear case monotonic: {is_mono}")
        
        # Test projection (only works for cubic)
        if degree == 3:
            try:
                projected = spline.project_to_monotonic(method="simple")
                print(f"   Degree {degree} projection successful")
            except Exception as e:
                print(f"   Degree {degree} projection failed: {e}")
        else:
            try:
                projected = spline.project_to_monotonic(method="simple")
                print(f"   ❌ Degree {degree} projection should have failed")
            except ValueError as e:
                print(f"   ✅ Degree {degree} projection correctly rejected: {e}")


def test_simple_projection_selective_increments():
    """Test that simple projection only adds increments where points are too close."""
    print("\n=== Testing Selective Increments in Simple Projection ===")
    
    # Test Case 1: Already monotonic points - should add no increments
    print("1. Already monotonic case:")
    monotonic_points = jnp.array([
        [0.0, 0.0],
        [0.5, 0.3], 
        [0.8, 0.7],
        [1.0, 1.0]
    ])
    
    mono_spline = BSpline(control_points=monotonic_points, degree=3)
    projected = mono_spline.project_to_monotonic(method="simple")
    
    print(f"  Original: {monotonic_points[:, 0]}")
    print(f"  Projected: {projected.control_points[:, 0]}")
    
    # Should be very close to original since already monotonic
    max_change = jnp.max(jnp.abs(projected.control_points - monotonic_points))
    print(f"  Max change: {max_change:.2e} (should be ~0)")
    
    # Test Case 2: Some tied points - should only add increments where needed
    print("\n2. Some tied points case:")
    tied_points = jnp.array([
        [0.0, 0.0],
        [0.5, 0.5],  # Same x as next point
        [0.5, 0.7],  # Tied with previous
        [1.0, 1.0]
    ])
    
    tied_spline = BSpline(control_points=tied_points, degree=3)
    tied_projected = tied_spline.project_to_monotonic(method="simple")
    
    print(f"  Original: {tied_points[:, 0]}")
    print(f"  Projected: {tied_projected.control_points[:, 0]}")
    
    # Should break the tie but keep other points mostly unchanged
    original_sorted = jnp.sort(tied_points[:, 0])
    projected_sorted = jnp.sort(tied_projected.control_points[:, 0])
    print(f"  Original sorted: {original_sorted}")
    print(f"  Projected sorted: {projected_sorted}")
    
    # Test Case 3: Multiple consecutive equal points
    print("\n3. Multiple consecutive equal points case:")
    multi_equal_points = jnp.array([
        [0.2, 0.0],
        [0.2, 0.3],  # Same x
        [0.2, 0.5],  # Same x
        [0.8, 1.0]
    ])
    
    multi_spline = BSpline(control_points=multi_equal_points, degree=3)
    multi_projected = multi_spline.project_to_monotonic(method="simple")
    
    print(f"  Original: {multi_equal_points[:, 0]}")
    print(f"  Projected: {multi_projected.control_points[:, 0]}")
    
    # Check that result is strictly monotonic
    x_coords = multi_projected.control_points[:, 0]
    is_strictly_increasing = jnp.all(x_coords[1:] > x_coords[:-1])
    print(f"  Is strictly increasing: {is_strictly_increasing}")
    
    # Test Case 4: Compare with old method (uniform increments)
    print("\n4. Comparison with uniform increment approach:")
    test_points = jnp.array([
        [0.1, 0.0],
        [0.6, 0.3],  # Already well-separated
        [0.6001, 0.5],  # Very close to previous (< epsilon)
        [0.9, 1.0]   # Already well-separated
    ])
    
    test_spline = BSpline(control_points=test_points, degree=3)
    selective_projected = test_spline.project_to_monotonic(method="simple")
    
    # Create uniform increment version for comparison
    sorted_test = jnp.sort(test_points, axis=0)
    n_control = sorted_test.shape[0]
    uniform_increments = jnp.arange(n_control) * 1e-6
    uniform_projected = sorted_test + uniform_increments[:, None]
    
    print(f"  Original: {test_points[:, 0]}")
    print(f"  Selective method: {selective_projected.control_points[:, 0]}")
    print(f"  Uniform increments: {uniform_projected[:, 0]}")
    
    # The selective method should preserve well-separated points better
    selective_change = jnp.sum(jnp.abs(selective_projected.control_points - sorted_test))
    uniform_change = jnp.sum(jnp.abs(uniform_projected - sorted_test))
    print(f"  Total change (selective): {selective_change:.2e}")
    print(f"  Total change (uniform): {uniform_change:.2e}")
    print(f"  Selective method preserves original better: {selective_change < uniform_change}")


def run_all_tests():
    """Run all monotonic tests."""
    print("🧪 Starting Comprehensive Monotonic Tests")
    print("=" * 60)
    
    test_monotonic_checking()
    test_monotonic_projection()
    test_projection_method_options()
    test_jax_compatibility()
    test_mathematical_conditions()
    test_edge_cases()
    test_simple_projection_selective_increments()
    
    print("\n" + "=" * 60)
    print("✅ All monotonic tests completed!")


if __name__ == "__main__":
    run_all_tests() 