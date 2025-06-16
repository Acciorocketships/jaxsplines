#!/usr/bin/env python3
"""
Test the new spline inversion functionality.
"""

import jax
import jax.numpy as jnp
from jaxsplines import BSpline

def test_spline_invert():
    """Test the analytical spline inversion method."""
    print("=== Testing B-spline Inversion ===")
    
    # Create a monotonic spline for testing
    control_points = jnp.array([
        [0.0, 0.0],   # Start at (0,0)
        [0.3, 1.0],   # Control point
        [0.7, 2.5],   # Control point  
        [1.0, 4.0]    # End at (1,4)
    ])
    
    spline = BSpline(control_points=control_points, degree=3)
    
    print(f"Control points shape: {control_points.shape}")
    print(f"Control points (x-dim): {control_points[:, 0]}")
    print(f"Control points (y-dim): {control_points[:, 1]}")
    
    # Test forward evaluation at known points
    test_params = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    forward_points = spline(test_params)
    
    print(f"\nForward evaluation:")
    for i, (t, point) in enumerate(zip(test_params, forward_points)):
        print(f"  t={t:.2f} -> ({point[0]:.4f}, {point[1]:.4f})")
    
    # Test inversion in x-dimension (should be monotonic)
    print(f"\n=== Testing X-dimension inversion ===")
    target_x_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for target_x in target_x_values:
        try:
            # Find t such that spline(t)[0] ≈ target_x
            inverted_t = spline.invert(target_x, dimension=0)
            
            # Verify by forward evaluation
            verification_point = spline(inverted_t)
            error_x = abs(verification_point[0] - target_x)
            
            print(f"  Target x={target_x:.3f} -> t={inverted_t:.6f}")
            print(f"    Verification: spline({inverted_t:.6f}) = ({verification_point[0]:.6f}, {verification_point[1]:.6f})")
            print(f"    Error in x: {error_x:.2e}")
            
            # Check if error is acceptably small
            if error_x < 1e-10:
                print(f"    ✅ Success!")
            else:
                print(f"    ⚠️  Large error")
                
        except Exception as e:
            print(f"  Target x={target_x:.3f} -> ERROR: {e}")
    
    # Test inversion in y-dimension
    print(f"\n=== Testing Y-dimension inversion ===")
    target_y_values = [0.5, 1.0, 2.0, 3.0, 3.5]
    
    for target_y in target_y_values:
        try:
            # Find t such that spline(t)[1] ≈ target_y
            inverted_t = spline.invert(target_y, dimension=1)
            
            # Verify by forward evaluation
            verification_point = spline(inverted_t)
            error_y = abs(verification_point[1] - target_y)
            
            print(f"  Target y={target_y:.3f} -> t={inverted_t:.6f}")
            print(f"    Verification: spline({inverted_t:.6f}) = ({verification_point[0]:.6f}, {verification_point[1]:.6f})")
            print(f"    Error in y: {error_y:.2e}")
            
            # Check if error is acceptably small
            if error_y < 1e-10:
                print(f"    ✅ Success!")
            else:
                print(f"    ⚠️  Large error")
                
        except Exception as e:
            print(f"  Target y={target_y:.3f} -> ERROR: {e}")
    
    # Test round-trip accuracy
    print(f"\n=== Testing Round-trip Accuracy ===")
    original_params = jnp.array([0.1, 0.3, 0.6, 0.9])
    
    for t_orig in original_params:
        # Forward: t -> point
        point = spline(t_orig)
        
        # Inverse: point[0] -> t (using x-coordinate)
        try:
            t_recovered = spline.invert(point[0], dimension=0)
            t_error = abs(t_recovered - t_orig)
            
            print(f"  Original t={t_orig:.3f} -> x={point[0]:.6f} -> recovered t={t_recovered:.6f}")
            print(f"    Round-trip error: {t_error:.2e}")
            
            if t_error < 1e-10:
                print(f"    ✅ Excellent round-trip!")
            elif t_error < 1e-6:
                print(f"    ✅ Good round-trip!")
            else:
                print(f"    ⚠️  Poor round-trip")
                
        except Exception as e:
            print(f"  Original t={t_orig:.3f} -> ERROR in round-trip: {e}")


def test_error_cases():
    """Test error handling in spline inversion."""
    print(f"\n=== Testing Error Cases ===")
    
    # Create a spline
    control_points = jnp.array([
        [0.0, 0.0],
        [0.5, 1.0], 
        [1.0, 2.0]
    ])
    spline = BSpline(control_points=control_points, degree=2)  # Quadratic
    
    # Test with non-cubic spline
    print("Testing non-cubic spline:")
    try:
        result = spline.invert(0.5, dimension=0)
        print(f"  ❌ Should have failed, but got: {result}")
    except ValueError as e:
        print(f"  ✅ Correctly rejected: {e}")
    
    # Test with cubic but non-monotonic spline
    print("\nTesting non-monotonic spline:")
    non_monotonic_points = jnp.array([
        [0.0, 0.0],
        [1.0, 2.0],  # Goes up
        [0.5, 1.0],  # Then down - not monotonic in x
        [2.0, 3.0]
    ])
    
    non_mono_spline = BSpline(control_points=non_monotonic_points, degree=3)
    
    try:
        result = non_mono_spline.invert(0.7, dimension=0, assume_monotonic=False)
        print(f"  ❌ Should have failed, but got: {result}")
    except ValueError as e:
        print(f"  ✅ Correctly rejected: {e}")
    
    # Test out-of-range target
    print("\nTesting out-of-range target:")
    monotonic_points = jnp.array([
        [0.0, 0.0],
        [0.3, 1.0],
        [0.7, 2.0],
        [1.0, 3.0]
    ])
    
    mono_spline = BSpline(control_points=monotonic_points, degree=3)
    
    try:
        result = mono_spline.invert(-0.5, dimension=0, assume_monotonic=False)  # Below minimum
        print(f"  ❌ Should have failed, but got: {result}")
    except ValueError as e:
        print(f"  ✅ Correctly rejected: {e}")


if __name__ == "__main__":
    test_spline_invert()
    test_error_cases()
    print("\n✅ Spline inversion tests completed!") 