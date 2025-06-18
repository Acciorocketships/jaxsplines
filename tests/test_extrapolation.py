#!/usr/bin/env python3
"""
Test the new extrapolation and improved inversion functionality.
"""

import jax.numpy as jnp
from jaxsplines import BSpline

def test_extrapolation():
    """Test B-spline extrapolation beyond [0,1]."""
    print("=== Testing B-spline Extrapolation ===")
    
    # Create a simple monotonic spline
    control_points = jnp.array([
        [0.0, 1.0],
        [0.5, 2.0], 
        [1.0, 3.0],
        [1.5, 4.0]
    ])
    
    spline = BSpline(control_points=control_points, degree=3)
    
    # Test extrapolation
    test_params = jnp.array([-0.5, -0.1, 0.0, 0.5, 1.0, 1.1, 1.5, 2.0])
    points = spline(test_params)
    
    print("Parameter -> Point")
    for t, point in zip(test_params, points):
        extrapolated = "📍" if 0 <= t <= 1 else "🚀"
        print(f"  t={t:4.1f} -> ({point[0]:6.3f}, {point[1]:6.3f}) {extrapolated}")
    
    # Test derivatives at boundaries
    print(f"\nDerivatives:")
    deriv_0 = spline.derivative(0.0, order=1)
    deriv_1 = spline.derivative(1.0, order=1)
    print(f"  At t=0: ({deriv_0[0]:.3f}, {deriv_0[1]:.3f})")
    print(f"  At t=1: ({deriv_1[0]:.3f}, {deriv_1[1]:.3f})")
    
    # Verify linear extrapolation
    point_neg = spline(-0.1)
    point_0 = spline(0.0)
    expected_neg = point_0 + (-0.1) * deriv_0
    
    print(f"\nLinear extrapolation verification:")
    print(f"  Actual  (-0.1): ({point_neg[0]:.6f}, {point_neg[1]:.6f})")
    print(f"  Expected(-0.1): ({expected_neg[0]:.6f}, {expected_neg[1]:.6f})")
    print(f"  Error: {jnp.linalg.norm(point_neg - expected_neg):.2e}")


def test_improved_inversion():
    """Test improved inversion with extrapolation support."""
    print(f"\n=== Testing Improved Inversion ===")
    
    # Create a monotonic spline (using projection)
    control_points = jnp.array([
        [0.0, 0.5],
        [0.3, 1.2],
        [0.8, 2.1], 
        [1.0, 3.0]
    ])
    
    spline = BSpline(control_points=control_points, degree=3)
    
    # Project to ensure monotonicity
    mono_spline = spline.project_to_monotonic(method="simple")
    
    print("Testing inversion within bounds:")
    
    # Test values within normal range
    test_targets = [0.6, 1.0, 1.5, 2.0, 2.5]
    
    for target_x in test_targets:
        try:
            t_inv = mono_spline.invert(target_x, dimension=0)
            point_check = mono_spline(t_inv)
            error = abs(point_check[0] - target_x)
            
            extrapolated = "🚀" if t_inv < 0 or t_inv > 1 else "📍"
            print(f"  Target x={target_x:.3f} -> t={t_inv:.6f} {extrapolated}")
            print(f"    Check: spline({t_inv:.6f})[0] = {point_check[0]:.6f}")
            print(f"    Error: {error:.2e}")
            
        except Exception as e:
            print(f"  Target x={target_x:.3f} -> ERROR: {e}")
    
    # Test extrapolation cases
    print(f"\nTesting extrapolation inversion:")
    
    # Get boundary values
    point_0 = mono_spline(0.0)
    point_1 = mono_spline(1.0)
    
    print(f"Boundary values: spline(0)=({point_0[0]:.3f}, {point_0[1]:.3f}), spline(1)=({point_1[0]:.3f}, {point_1[1]:.3f})")
    
    # Test beyond boundaries
    extreme_targets = [point_0[0] - 0.2, point_1[0] + 0.3]  # Beyond normal range
    
    for target_x in extreme_targets:
        try:
            t_inv = mono_spline.invert(target_x, dimension=0)
            point_check = mono_spline(t_inv)
            error = abs(point_check[0] - target_x)
            
            print(f"  Extreme x={target_x:.3f} -> t={t_inv:.6f} 🚀")
            print(f"    Check: spline({t_inv:.6f})[0] = {point_check[0]:.6f}")
            print(f"    Error: {error:.2e}")
            
        except Exception as e:
            print(f"  Extreme x={target_x:.3f} -> ERROR: {e}")


if __name__ == "__main__":
    test_extrapolation()
    test_improved_inversion()
    print("\n✅ Extrapolation and inversion tests completed!") 