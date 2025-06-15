import jax
import jax.numpy as jnp
from jax import vmap
import equinox as eqx
from typing import Optional, Union, Tuple
from jaxtyping import Array, Float


class BSpline(eqx.Module):
    """
    B-spline implementation using JAX and Equinox with learnable parameters.
    
    This implementation supports:
    - Learnable control points
    - Vectorization with vmap
    - Polynomial degrees 1-3 (linear, quadratic, cubic)
    - UNIFORM knot spacing only (knot vector is not stored)
    - Monotonicity constraints for training
    """
    
    control_points: Float[Array, "n_control_points dim"] 
    degree: int = eqx.static_field()
    
    def __init__(
        self,
        control_points: Float[Array, "n_control_points dim"],
        degree: int = 3,
        key: Optional[jax.random.PRNGKey] = None
    ):
        """
        Initialize B-spline with control points.
        
        Args:
            control_points: Control points defining the spline shape
            degree: Polynomial degree of the B-spline
            key: Random key for initialization (unused)
            
        Note:
            This implementation assumes uniform knot spacing.
        """
        # Validate degree
        if degree < 1 or degree > 3:
            raise ValueError(f"Only degrees 1-3 are supported, got degree {degree}")
        
        # Validate minimum control points
        n_control_points = control_points.shape[0]
        min_control_points = degree + 1
        if n_control_points < min_control_points:
            raise ValueError(
                f"Need at least {min_control_points} control points for degree {degree}, "
                f"got {n_control_points}"
            )
        
        self.control_points = control_points
        self.degree = degree
    
    def _compute_basis_functions(self, t: Float[Array, ""]) -> Float[Array, "n_control_points"]:
        """
        Compute all B-spline basis functions at parameter t.
        Assumes uniform knot spacing.
        """
        n_control_points = self.control_points.shape[0]
        
        # Clamp t to valid range
        t = jnp.clip(t, 0.0, 1.0)
        
        # Use JAX switch for clean degree selection
        branches = [
            lambda: self._compute_linear_basis(t, n_control_points),     # degree 0 (fallback to degree 1)
            lambda: self._compute_linear_basis(t, n_control_points),     # degree 1
            lambda: self._compute_quadratic_basis(t, n_control_points),  # degree 2
            lambda: self._compute_cubic_basis(t, n_control_points),      # degree 3
        ]
        
        # Clamp degree to valid range [1, 3]
        degree_index = jnp.clip(self.degree, 1, 3)
        
        return jax.lax.switch(degree_index, branches)
    
    def _compute_cubic_basis(self, t: Float[Array, ""], n_control_points: int) -> Float[Array, "n_control_points"]:
        """Compute cubic B-spline basis functions with uniform knot spacing."""
        # Scale parameter to segment space
        t_scaled = t * (n_control_points - 3)
        
        # Find the segment
        segment = jnp.floor(t_scaled).astype(int)
        segment = jnp.clip(segment, 0, n_control_points - 4)
        
        # Local parameter within segment
        u = t_scaled - segment
        
        # Cubic B-spline basis functions
        b0 = (1 - u) ** 3 / 6
        b1 = (3 * u ** 3 - 6 * u ** 2 + 4) / 6
        b2 = (-3 * u ** 3 + 3 * u ** 2 + 3 * u + 1) / 6
        b3 = u ** 3 / 6
        
        # Create result array
        result = jnp.zeros(n_control_points)
        i_indices = jnp.arange(n_control_points)
        
        # Place basis functions at correct positions
        def add_basis_contribution(result, basis_val, offset):
            pos = segment + offset
            mask = (i_indices == pos) & (pos >= 0) & (pos < n_control_points)
            return result + basis_val * mask
        
        result = add_basis_contribution(result, b0, 0)
        result = add_basis_contribution(result, b1, 1)
        result = add_basis_contribution(result, b2, 2)
        result = add_basis_contribution(result, b3, 3)
        
        return result
    
    def _compute_quadratic_basis(self, t: Float[Array, ""], n_control_points: int) -> Float[Array, "n_control_points"]:
        """Compute quadratic B-spline basis functions with uniform knot spacing."""
        t_scaled = t * (n_control_points - 2)
        segment = jnp.floor(t_scaled).astype(int)
        segment = jnp.clip(segment, 0, n_control_points - 3)
        u = t_scaled - segment
        
        # Quadratic B-spline basis functions
        b0 = (1 - u) ** 2 / 2
        b1 = (-2 * u ** 2 + 2 * u + 1) / 2
        b2 = u ** 2 / 2
        
        result = jnp.zeros(n_control_points)
        i_indices = jnp.arange(n_control_points)
        
        def add_basis_contribution(result, basis_val, offset):
            pos = segment + offset
            mask = (i_indices == pos) & (pos >= 0) & (pos < n_control_points)
            return result + basis_val * mask
        
        result = add_basis_contribution(result, b0, 0)
        result = add_basis_contribution(result, b1, 1)
        result = add_basis_contribution(result, b2, 2)
        
        return result
    
    def _compute_linear_basis(self, t: Float[Array, ""], n_control_points: int) -> Float[Array, "n_control_points"]:
        """Compute linear B-spline basis functions with uniform knot spacing."""
        t_scaled = t * (n_control_points - 1)
        segment = jnp.floor(t_scaled).astype(int)
        segment = jnp.clip(segment, 0, n_control_points - 2)
        u = t_scaled - segment
        
        # Linear B-spline basis functions
        b0 = 1 - u
        b1 = u
        
        result = jnp.zeros(n_control_points)
        i_indices = jnp.arange(n_control_points)
        
        def add_basis_contribution(result, basis_val, offset):
            pos = segment + offset
            mask = (i_indices == pos) & (pos >= 0) & (pos < n_control_points)
            return result + basis_val * mask
        
        result = add_basis_contribution(result, b0, 0)
        result = add_basis_contribution(result, b1, 1)
        
        return result
    
    def __call__(self, t: Union[float, Float[Array, "..."]]) -> Float[Array, "... dim"]:
        """
        Evaluate the B-spline at parameter value(s) t.
        
        Args:
            t: Parameter value(s) where to evaluate the spline
            
        Returns:
            Point(s) on the spline curve
        """
        t = jnp.asarray(t)
        original_shape = t.shape
        t_flat = t.ravel()
        
        def evaluate_single(t_val):
            # Get basis function values
            basis_values = self._compute_basis_functions(t_val)
            
            # Compute weighted sum of control points
            return jnp.sum(basis_values[:, None] * self.control_points, axis=0)
        
        # Vectorize over parameter values
        results = vmap(evaluate_single)(t_flat)
        
        # Reshape to match input shape + dimension
        return results.reshape(original_shape + (self.control_points.shape[1],))
    
    def derivative(self, t: Union[float, Float[Array, "..."]], order: int = 1) -> Float[Array, "... dim"]:
        """
        Compute the derivative of the B-spline at parameter value(s) t.
        Uses uniform knot spacing assumption for derivative computation.
        
        Args:
            t: Parameter value(s) where to evaluate the derivative
            order: Order of the derivative (1 for first derivative, etc.)
            
        Returns:
            Derivative vector(s) at the given parameter value(s)
        """
        if order <= 0:
            return self(t)
        
        if self.degree < order:
            # Derivative order higher than spline degree results in zero
            t = jnp.asarray(t)
            return jnp.zeros(t.shape + (self.control_points.shape[1],))
        
        # Compute derivative control points using uniform knot spacing
        n_control_points = self.control_points.shape[0]
        derivative_control_points = self.control_points.copy()
        current_degree = self.degree
        
        for _ in range(order):
            new_control_points = []
            for i in range(n_control_points - 1):
                # For uniform knots, the knot spacing is 1/(n_control_points - current_degree)
                # So the coefficient is current_degree * (n_control_points - current_degree)
                coeff = current_degree * (n_control_points - current_degree)
                new_point = coeff * (derivative_control_points[i + 1] - derivative_control_points[i])
                new_control_points.append(new_point)
            
            derivative_control_points = jnp.stack(new_control_points)
            current_degree -= 1
            n_control_points -= 1
        
        # Create derivative spline
        derivative_spline = BSpline(
            control_points=derivative_control_points,
            degree=current_degree
        )
        
        return derivative_spline(t)
    
    def project_to_monotonic(self, epsilon: float = 1e-6) -> 'BSpline':
        """
        Project control points to satisfy monotonicity constraints in ALL dimensions.
        JAX-compatible version using functional programming.
        
        Args:
            epsilon: Small value to ensure strict inequalities
            
        Returns:
            New BSpline with projected control points that are monotonic in all dimensions
        """
        # Validate degree (this happens before JIT compilation, so it's safe)
        if self.degree != 3:
            raise ValueError("Monotonicity projection currently only implemented for cubic B-splines")
        
        return self._project_monotonic_impl(epsilon)
    
    def _project_monotonic_impl(self, epsilon: float) -> 'BSpline':
        """
        Implementation of monotonic projection (JAX-compatible, no degree validation).
        """
        new_control_points = self.control_points.copy()
        n_segments = self.control_points.shape[0] - 3
        n_dimensions = self.control_points.shape[1]
        
        if n_segments <= 0:
            return BSpline(control_points=new_control_points, degree=self.degree)
        
        return self._project_all_dimensions(new_control_points, n_segments, n_dimensions, epsilon)
             
    def _project_all_dimensions(self, new_control_points, n_segments, n_dimensions, epsilon):
        """Helper method to project all dimensions (extracted for JAX compatibility)."""
        # Apply projection to each dimension (vectorized over segments within each dimension)
        for dimension in range(n_dimensions):
            # Extract all segment control points for this dimension at once
            segment_indices = jnp.arange(n_segments)[:, None] + jnp.arange(4)[None, :]
            segment_points = new_control_points[segment_indices, dimension]  # Shape: (n_segments, 4)
            
            # Vectorized projection across all segments in this dimension
            def project_all_segments(segment_points):
                p0, p1, p2, p3 = segment_points[:, 0], segment_points[:, 1], segment_points[:, 2], segment_points[:, 3]
                
                # For each segment, choose between discriminant and vertex-outside projections
                def project_single_segment(p0_i, p1_i, p2_i, p3_i):

                    # Enforce P2 > P0 + epsilon and P3 > P1 + epsilon
                    p2_proj_i = jnp.maximum(p2_i, p0_i + epsilon)
                    p3_proj_i = jnp.maximum(p3_i, p1_i + epsilon)

                    # Store original
                    orig = jnp.array([p0_i, p1_i, p2_proj_i, p3_proj_i])
                    
                    # Try discriminant projection
                    discriminant_result = self._project_discriminant(p0_i, p1_i, p2_proj_i, p3_proj_i, epsilon)
                    discriminant_diff = jnp.linalg.norm(discriminant_result - orig)
                    
                    # Try vertex-outside projection  
                    vertex_result = self._project_vertex_outside(p0_i, p1_i, p2_proj_i, p3_proj_i, epsilon)
                    vertex_diff = jnp.linalg.norm(vertex_result - orig)
                    
                    # Choose the projection with smaller change
                    use_discriminant = discriminant_diff <= vertex_diff
                    
                    return jnp.where(
                        use_discriminant,
                        discriminant_result,
                        vertex_result
                    )
                
                # Vectorize over all segments
                projected_segments = jax.vmap(project_single_segment)(p0, p1, p2, p3)
                return projected_segments
            
            # Apply projection and update control points for this dimension
            projected_segments = project_all_segments(segment_points)
            new_control_points = new_control_points.at[segment_indices, dimension].set(projected_segments)
        
        return BSpline(
            control_points=new_control_points,
            degree=self.degree
        )
    
    def project_to_monotonic_simple(self, epsilon: float = 1e-6) -> 'BSpline':
        """
        Fully vectorized JAX-compatible monotonicity projection that makes sure the knots are in increasing order.
        """
        # Validate degree (this happens before JIT compilation, so it's safe)
        if self.degree != 3:
            raise ValueError("Monotonicity projection currently only implemented for cubic B-splines")
        
        return self._project_monotonic_simple_impl(epsilon)
    
    def _project_monotonic_simple_impl(self, epsilon: float) -> 'BSpline':
        """
        Simple implementation: just sort control points in each dimension and add increments.
        """
        # Sort control points in each dimension and add small increments for strict ordering
        sorted_points = jnp.sort(self.control_points, axis=0)  # Sort along control points axis
        
        # Add increments to ensure strict monotonicity: [0, ε, 2ε, 3ε, ...]
        n_control_points = sorted_points.shape[0]
        increments = jnp.arange(n_control_points) * epsilon
        
        # Broadcast increments to all dimensions
        monotonic_points = sorted_points + increments[:, None]
        
        return BSpline(
            control_points=monotonic_points,
            degree=self.degree
        )
    
    def _project_discriminant(self, p0, p1, p2, p3, epsilon):
        """
        Project control points to satisfy discriminant condition:
        (P₀ - 2P₁ + P₂)² < (P₂ - P₀)(-P₀ + 3P₁ - 3P₂ + P₃)
        
        Strategy: Adjust P₃ to make the RHS larger than LHS + epsilon.
        If denominator is zero, division gives inf, making this projection unattractive.
        """
        lhs = (p0 - 2*p1 + p2)**2
        # We want: lhs < (p2 - p0) * (-p0 + 3*p1 - 3*p2 + p3_new)
        # Solve for p3_new: p3_new > (lhs / (p2 - p0)) + p0 - 3*p1 + 3*p2
        
        denominator = p2 - p0
        
        # Let division by zero naturally produce inf, which makes this projection unattractive
        p3_new = jnp.maximum(p3, (lhs / denominator) + p0 - 3*p1 + 3*p2 + epsilon)
        
        return jnp.array([p0, p1, p2, p3_new])
    
    def _project_vertex_outside(self, p0, p1, p2, p3, epsilon):
        """
        Project control points to satisfy vertex-outside conditions.
        
        Args:
            use_left: If True, use "left of 0" case, otherwise "right of 1" case
        """
        orig = jnp.array([p0, p1, p2, p3])
        
        # Common condition for both cases: P₃ > P₀ - 3P₁ + 3P₂
        p3_min_common = p0 - 3*p1 + 3*p2 + epsilon
        p3_proj = jnp.maximum(p3, p3_min_common)
        
        # Left of 0 case: P₀ ≥ 2P₁ - P₂ and P₃ > P₀ - 3P₁ + 3P₂
        p0_min = 2*p1 - p2
        p0_proj_left = jnp.maximum(p0, p0_min)
        left_result = jnp.array([p0_proj_left, p1, p2, p3_proj])
        
        # Right of 1 case: P₂ ≥ (P₃ + P₁) / 2 and P₃ > P₀ - 3P₁ + 3P₂
        p2_min = 0.5 * (p3 + p1)
        p2_proj_right = jnp.maximum(p2, p2_min)
        right_result = jnp.array([p0, p1, p2_proj_right, p3_proj])

        right_diff = jnp.linalg.norm(right_result - orig)
        left_diff = jnp.linalg.norm(left_result - orig)

        use_left = left_diff < right_diff

        # Choose between left and right projections
        return jnp.where(use_left, left_result, right_result)