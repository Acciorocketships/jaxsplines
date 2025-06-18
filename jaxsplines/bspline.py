import jax
import jax.numpy as jnp
from jax import vmap
import equinox as eqx
from typing import Optional, Union
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
    
    control_points: Float[Array, "n_control_points dim"]  # type: ignore
    degree: int = eqx.static_field()
    
    def __init__(
        self,
        control_points: Float[Array, "n_control_points dim"],  # type: ignore
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
    
    def _compute_basis_functions(self, t: Float[Array, ""]) -> Float[Array, "n_control_points"]:  # type: ignore
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
    
    def _compute_cubic_basis(self, t: Float[Array, ""], n_control_points: int) -> Float[Array, "n_control_points"]:  # type: ignore
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
    
    def _compute_quadratic_basis(self, t: Float[Array, ""], n_control_points: int) -> Float[Array, "n_control_points"]:  # type: ignore
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
    
    def _compute_linear_basis(self, t: Float[Array, ""], n_control_points: int) -> Float[Array, "n_control_points"]:  # type: ignore
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
    
    def __call__(self, t: Union[float, Float[Array, "..."]]) -> Float[Array, "... dim"]:  # type: ignore
        """
        Evaluate the B-spline at parameter value(s) t.
        Supports extrapolation beyond [0,1] using linear extension.
        
        Args:
            t: Parameter value(s) where to evaluate the spline
            
        Returns:
            Point(s) on the spline curve
        """
        t = jnp.asarray(t)
        original_shape = t.shape
        t_flat = t.ravel()
        
        def evaluate_single(t_val):
            # Handle extrapolation outside [0,1] using JAX conditionals
            # Linear extrapolation to -∞ using derivative at t=0
            point_0 = self._evaluate_core(0.0)
            deriv_0 = self._derivative_core(0.0, order=1)
            extrap_neg = point_0 + t_val * deriv_0
            
            # Linear extrapolation to +∞ using derivative at t=1  
            point_1 = self._evaluate_core(1.0)
            deriv_1 = self._derivative_core(1.0, order=1)
            extrap_pos = point_1 + (t_val - 1.0) * deriv_1
            
            # Normal evaluation within [0,1]
            normal_eval = self._evaluate_core(t_val)
            
            # Use JAX conditionals instead of Python if statements
            return jnp.where(
                t_val < 0,
                extrap_neg,
                jnp.where(t_val > 1, extrap_pos, normal_eval)
            )
        
        # Vectorize over parameter values
        results = vmap(evaluate_single)(t_flat)
        
        # Reshape to match input shape + dimension
        return results.reshape(original_shape + (self.control_points.shape[1],))
    
    def _evaluate_core(self, t_val: float) -> jnp.ndarray:
        """
        Core B-spline evaluation within the natural domain [0,1].
        
        This is the internal implementation that computes the mathematical B-spline
        evaluation using basis functions. It does NOT handle extrapolation - that's
        handled by the public __call__ method which uses this function for values
        within [0,1] and implements linear extrapolation for values outside.
        
        Args:
            t_val: Parameter value in [0,1] where to evaluate the spline
            
        Returns:
            Point vector at t_val using the B-spline basis function formulation
        """
        # Get basis function values
        basis_values = self._compute_basis_functions(t_val)
        
        # Compute weighted sum of control points
        return jnp.sum(basis_values[:, None] * self.control_points, axis=0)
    

    
    def _derivative_core(self, t_val: float, order: int = 1) -> jnp.ndarray:
        """
        Core B-spline derivative computation within the natural domain [0,1].
        
        This is the internal implementation that computes the mathematical derivative
        of the B-spline basis functions. It does NOT handle extrapolation - that's
        handled by the public derivative() method which calls this function to get
        boundary slopes for linear extrapolation.
        
        Args:
            t_val: Parameter value in [0,1] where to evaluate derivative
            order: Order of derivative (1=first derivative, 2=second derivative, etc.)
            
        Returns:
            Derivative vector at t_val using the B-spline basis derivative formulation
        """
        if order <= 0:
            return self._evaluate_core(t_val)
        
        if self.degree < order:
            # Derivative order higher than spline degree results in zero
            return jnp.zeros(self.control_points.shape[1])
        
        # Compute derivative control points using uniform knot spacing
        n_control_points = self.control_points.shape[0]
        derivative_control_points = self.control_points.copy()
        current_degree = self.degree
        
        for _ in range(order):
            new_control_points = []
            for i in range(n_control_points - 1):
                coeff = current_degree * (n_control_points - current_degree)
                new_point = coeff * (derivative_control_points[i + 1] - derivative_control_points[i])
                new_control_points.append(new_point)
            
            derivative_control_points = jnp.stack(new_control_points)
            current_degree -= 1
            n_control_points -= 1
        
        # Create derivative spline and evaluate within [0,1] domain
        derivative_spline = BSpline(
            control_points=derivative_control_points,
            degree=current_degree
        )
        
        return derivative_spline._evaluate_core(t_val)
    
    def derivative(self, t: Union[float, Float[Array, "..."]], order: int = 1) -> Float[Array, "... dim"]:  # type: ignore
        """
        Compute the derivative of the B-spline at parameter value(s) t.
        
        This method handles extrapolation by using boundary derivatives:
        - For t < 0: returns derivative at t=0 (constant extrapolation of slope)
        - For t > 1: returns derivative at t=1 (constant extrapolation of slope)  
        - For t ∈ [0,1]: returns the actual B-spline derivative
        
        This ensures that the extrapolated regions have constant slopes matching
        the boundary behavior, which is consistent with linear extrapolation.
        
        Args:
            t: Parameter value(s) where to evaluate the derivative
            order: Order of the derivative (1 for first derivative, etc.)
            
        Returns:
            Derivative vector(s) at the given parameter value(s)
        """
        t = jnp.asarray(t)
        original_shape = t.shape
        t_flat = t.ravel()
        
        def derivative_single(t_val):
            # For extrapolated regions, use boundary derivatives
            deriv_0 = self._derivative_core(0.0, order)
            deriv_1 = self._derivative_core(1.0, order)
            deriv_normal = self._derivative_core(t_val, order)
            
            # Use JAX conditionals
            return jnp.where(
                t_val < 0,
                deriv_0,
                jnp.where(t_val > 1, deriv_1, deriv_normal)
            )
        
        # Vectorize over parameter values
        results = vmap(derivative_single)(t_flat)
        
        # Reshape to match input shape + dimension
        return results.reshape(original_shape + (self.control_points.shape[1],))
    
    def invert(self, target_values: Union[float, Float[Array, "..."]], dimension: int = 0, 
               tolerance: float = 1e-12, assume_monotonic: bool = True) -> Union[float, Float[Array, "..."]]:  # type: ignore
        """
        Find the parameter t that gives the target_values in the specified dimension.
        Uses analytical inversion via Cardano's formula for monotonic cubic B-splines.
        
        Works with both scalar and array inputs automatically.
        
        Args:
            target_values: The desired output value(s) to find t for (scalar or array)
            dimension: Which dimension to invert (0 for x, 1 for y, etc.)
            tolerance: Numerical tolerance for degenerate cases
            assume_monotonic: If True, skip monotonicity validation (use after project_to_monotonic)
            
        Returns:
            Parameter value(s) t such that spline(t)[dimension] ≈ target_values
            Shape matches input: scalar -> scalar, array -> array
            
        Raises:
            ValueError: If spline is not cubic or fails monotonicity check
        """
        # Static checks that don't interfere with JAX tracing
        if self.degree != 3:
            raise ValueError("Analytical inversion only implemented for cubic B-splines")
        
        # Rigorous monotonicity check unless explicitly disabled
        if not assume_monotonic:
            if not self.check_monotonic(dimension, tolerance):
                raise ValueError(f"Spline is not monotonic in dimension {dimension}. "
                               f"Use project_to_monotonic() first, or set assume_monotonic=True if you're sure.")
        
        target_values = jnp.asarray(target_values)
        original_shape = target_values.shape
        
        # For scalar inputs, use the core implementation directly
        if target_values.ndim == 0:
            return self._invert_core(float(target_values), dimension, tolerance)
        
        # For array inputs, use vmap to vectorize over the target values
        from functools import partial
        invert_core_partial = partial(self._invert_core, dimension=dimension, tolerance=tolerance)
        
        # Vectorize over the flattened input
        target_flat = target_values.ravel()
        results_flat = vmap(invert_core_partial)(target_flat)
        
        # Reshape to match input shape
        return results_flat.reshape(original_shape)
    

    
    def _invert_core(self, target_value: float, dimension: int, tolerance: float) -> float:
        """
        Core JAX-compatible invert implementation.
        Handles extrapolation and normal inversion using analytical methods.
        """
        # Get boundary values and derivatives for extrapolation check
        point_0 = self._evaluate_core(0.0)[dimension]
        point_1 = self._evaluate_core(1.0)[dimension] 
        min_boundary = jnp.minimum(point_0, point_1)
        max_boundary = jnp.maximum(point_0, point_1)
        
        # Extrapolation to negative t values (target < min_boundary)
        deriv_0 = self._derivative_core(0.0, order=1)[dimension]
        t_extrap_neg = jnp.where(
            jnp.abs(deriv_0) < tolerance,
            0.0,  # Derivative too small, return boundary
            (target_value - point_0) / deriv_0
        )
        
        # Extrapolation to t > 1 values (target > max_boundary)
        deriv_1 = self._derivative_core(1.0, order=1)[dimension]
        t_extrap_pos = jnp.where(
            jnp.abs(deriv_1) < tolerance,
            1.0,  # Derivative too small, return boundary
            1.0 + (target_value - point_1) / deriv_1
        )
        
        # Normal inversion within bounds [0,1]
        # Find which span contains the target value
        n_control_points = self.control_points.shape[0]
        n_spans = n_control_points - 3
        
        # Evaluate spline at span boundaries to find the right span
        span_params = jnp.linspace(0, 1, n_spans + 1)
        span_values = self(span_params)[:, dimension]
        
        # Find the span where target_value lies
        span_idx = jnp.searchsorted(span_values[:-1], target_value, side='right') - 1
        span_idx = jnp.clip(span_idx, 0, n_spans - 1)
        
        # Get the four control points for this span using dynamic slicing
        control_points_dim = self.control_points[:, dimension]
        from jax import lax
        control_points_span = lax.dynamic_slice(control_points_dim, (span_idx,), (4,))
        
        # Convert to local parameter and invert analytically
        u_local = self._invert_span_analytic(target_value, control_points_span, tolerance)
        
        # Convert back to global parameter
        t_normal = (span_idx + u_local) / n_spans
        t_normal = jnp.clip(t_normal, 0.0, 1.0)
        
        # Use JAX conditionals to select the appropriate result
        result = jnp.where(
            target_value < min_boundary,
            t_extrap_neg,
            jnp.where(target_value > max_boundary, t_extrap_pos, t_normal)
        )
        
        return result
    
    def _invert_span_analytic(self, y: float, P: jnp.ndarray, tolerance: float = 1e-12) -> float:
        """
        Analytically invert one cubic B-spline span using Cardano's formula.
        JAX-compatible version using jnp.where instead of Python conditionals.
        
        Args:
            y: Target ordinate value
            P: Array of 4 control point values [P0, P1, P2, P3] for this span
            tolerance: Numerical tolerance for degenerate cases
            
        Returns:
            Local parameter u in [0,1] such that the span evaluates to y
        """
        P0, P1, P2, P3 = P[0], P[1], P[2], P[3]
        
        # Convert B-spline basis to standard cubic form: A*u³ + B*u² + C*u + D
        A = (-P0 + 3*P1 - 3*P2 + P3) / 6
        B = (P0 - 2*P1 + P2) / 2
        C = (P2 - P0) / 2
        D = (P0 + 4*P1 + P2) / 6
        F = D - y  # Constant term after moving y to left side
        
        # Handle degenerate cases using JAX conditionals
        # Linear case: C*u + F = 0
        u_linear = jnp.where(
            jnp.abs(C) < tolerance,
            0.5,  # Fallback value if coefficient is also zero
            jnp.clip(-F / C, 0.0, 1.0)
        )
        
        # Quadratic case: B*u² + C*u + F = 0
        discriminant = C*C - 4*B*F
        sqrt_disc = jnp.sqrt(jnp.maximum(discriminant, 0.0))  # Ensure non-negative
        u1_quad = (-C + sqrt_disc) / (2*B)
        u2_quad = (-C - sqrt_disc) / (2*B)
        
        # Choose the quadratic root that's closest to [0,1]
        u1_valid = (u1_quad >= 0) & (u1_quad <= 1)
        u2_valid = (u2_quad >= 0) & (u2_quad <= 1)
        
        u_quad = jnp.where(
            u1_valid,
            u1_quad,
            jnp.where(u2_valid, u2_quad, jnp.clip(u1_quad, 0.0, 1.0))
        )
        
        # Full cubic case - depress the cubic
        # Transform to v³ + p*v + q = 0 with u = v - B/(3*A)
        p = (3*A*C - B*B) / (3*A*A)
        q = (2*B*B*B - 9*A*B*C + 27*A*A*F) / (27*A*A*A)
        
        # Cardano's formula
        Delta = (q/2)**2 + (p/3)**3
        
        # One real root case (Delta >= 0)
        sqrt_delta = jnp.sqrt(jnp.maximum(Delta, 0.0))
        term1 = -q/2 + sqrt_delta
        term2 = -q/2 - sqrt_delta
        
        cbrt1 = jnp.sign(term1) * jnp.abs(term1)**(1/3)
        cbrt2 = jnp.sign(term2) * jnp.abs(term2)**(1/3)
        
        v_single = cbrt1 + cbrt2
        u_cubic_single = v_single - B/(3*A)
        
        # Three real roots case (Delta < 0)
        r = 2 * jnp.sqrt(-p/3)
        phi = jnp.arccos(jnp.clip(-q/2 / jnp.sqrt(-(p/3)**3), -1.0, 1.0))
        
        u0 = r * jnp.cos(phi/3) - B/(3*A)
        u1 = r * jnp.cos((phi + 2*jnp.pi)/3) - B/(3*A)
        u2 = r * jnp.cos((phi + 4*jnp.pi)/3) - B/(3*A)
        
        # Choose the root closest to [0,1]
        roots = jnp.array([u0, u1, u2])
        valid_mask = (roots >= 0) & (roots <= 1)
        
        # If we have valid roots, choose the first valid one
        # Otherwise, choose the root closest to [0,1]
        valid_roots = jnp.where(valid_mask, roots, jnp.inf)
        distances = jnp.minimum(jnp.abs(roots), jnp.abs(roots - 1))
        
        u_cubic_triple = jnp.where(
            jnp.any(valid_mask),
            jnp.min(valid_roots),
            jnp.clip(roots[jnp.argmin(distances)], 0.0, 1.0)
        )
        
        # Choose between single and triple root cases based on Delta
        u_cubic = jnp.where(Delta >= 0, u_cubic_single, u_cubic_triple)
        u_cubic = jnp.clip(u_cubic, 0.0, 1.0)
        
        # Select the appropriate case based on coefficient magnitudes
        # Use cubic if A is significant, quadratic if B is significant, otherwise linear
        result = jnp.where(
            jnp.abs(A) >= tolerance,
            u_cubic,
            jnp.where(jnp.abs(B) >= tolerance, u_quad, u_linear)
        )
        
        return result
    
    def project_to_monotonic(self, method: str = "simple", epsilon: float = 1e-3) -> 'BSpline':
        """
        Project control points to satisfy monotonicity constraints in ALL dimensions.
        
        Args:
            method: Projection method to use:
                - "simple": Sort control points and add increments (fast, vectorized)
                - "exact": Use mathematical conditions for minimal projection (slower, more precise)
            epsilon: Small value to ensure strict inequalities between control points
            
        Returns:
            New BSpline with projected control points that are monotonic in all dimensions
            
        Raises:
            ValueError: If degree is not 3 (cubic splines only) or invalid method
        """
        # Validate degree (this happens before JIT compilation, so it's safe)
        if self.degree != 3:
            raise ValueError("Monotonicity projection currently only implemented for cubic B-splines")
        
        if method not in ["simple", "exact"]:
            raise ValueError(f"Method must be 'simple' or 'exact', got '{method}'")
        
        if method == "simple":
            # Simple method: sort control points and only add increments where needed
            sorted_points = jnp.sort(self.control_points, axis=0)  # Sort along control points axis
            
            # Vectorized processing for all dimensions at once
            # Find where consecutive points are too close (difference < epsilon) across all dimensions
            diffs = jnp.diff(sorted_points, axis=0)  # Shape: (n_control_points-1, n_dimensions)
            too_close_mask = diffs < epsilon
            
            # For points that are too close, compute the increment needed
            increment_needed = jnp.where(too_close_mask, epsilon - diffs, 0.0)
            
            # Pad with zeros at the beginning (first point doesn't need increment from previous)
            # Shape: (n_control_points, n_dimensions)
            increment_needed_padded = jnp.concatenate([
                jnp.zeros((1, sorted_points.shape[1])), 
                increment_needed
            ], axis=0)
            
            # Cumulative sum gives us the total increment needed at each point for each dimension
            cumulative_increments = jnp.cumsum(increment_needed_padded, axis=0)
            
            # Apply increments to all dimensions at once
            monotonic_points = sorted_points + cumulative_increments
            
            return BSpline(
                control_points=monotonic_points,
                degree=self.degree
            )
        
        else:  # method == "exact"
            # Exact method: use mathematical conditions for minimal projection
            new_control_points = self.control_points.copy()
            n_segments = self.control_points.shape[0] - 3
            n_dimensions = self.control_points.shape[1]
            
            if n_segments <= 0:
                return BSpline(control_points=new_control_points, degree=self.degree)
            
            return self._project_all_dimensions(new_control_points, n_segments, n_dimensions, epsilon)
             
    def _project_all_dimensions(self, new_control_points, n_segments, n_dimensions, epsilon):
        """Helper method to project all dimensions using exact mathematical conditions."""
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
        
        Chooses between "left of 0" and "right of 1" projections based on 
        which produces smaller changes to the original control points.
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
    
    def check_monotonic(self, dimension: int = 0, epsilon: float = 1e-12) -> bool:
        """
        Check if the spline is monotonic in the specified dimension using mathematical conditions.
        
        Fixed version that handles edge cases in the discriminant condition properly.
        
        Args:
            dimension: Which dimension to check (0 for x, 1 for y, etc.)
            epsilon: Tolerance for numerical precision
            
        Returns:
            True if spline is monotonic in the specified dimension, False otherwise
        """
        if self.degree != 3:
            # For non-cubic splines, fall back to simple control point check
            control_values = self.control_points[:, dimension]
            is_increasing = jnp.all(control_values[1:] >= control_values[:-1] - epsilon)
            is_decreasing = jnp.all(control_values[1:] <= control_values[:-1] + epsilon)
            return is_increasing or is_decreasing
        
        n_control_points = self.control_points.shape[0]
        n_segments = n_control_points - 3
        
        if n_segments <= 0:
            return True
        
        # Check each segment (span) for monotonicity conditions
        for seg_idx in range(n_segments):
            # Get the four control points for this segment
            P = self.control_points[seg_idx:seg_idx + 4, dimension]
            P0, P1, P2, P3 = P[0], P[1], P[2], P[3]
            
            # Basic ordering check: P₂ >= P₀ and P₃ >= P₁ (for increasing)
            # or P₂ <= P₀ and P₃ <= P₁ (for decreasing)
            increasing_basic = (P2 >= P0 - epsilon) and (P3 >= P1 - epsilon)
            decreasing_basic = (P2 <= P0 + epsilon) and (P3 <= P1 + epsilon)
            
            if not (increasing_basic or decreasing_basic):
                return False
            
            # Check if this segment is monotonic using corrected conditions
            if not self._check_segment_monotonic(P0, P1, P2, P3, epsilon):
                return False
        
        return True
    
    def _check_segment_monotonic(self, P0: float, P1: float, P2: float, P3: float, epsilon: float) -> bool:
        """
        Check if a single segment is monotonic using corrected mathematical conditions.
        
        For monotonicity, the derivative should not change sign in [0,1].
        """
        # Coefficients of the derivative (quadratic): A*u² + B*u + C
        A = (-P0 + 3*P1 - 3*P2 + P3) / 2  # Coefficient of u²
        B = (P0 - 2*P1 + P2)              # Coefficient of u
        C = (P2 - P0) / 2                 # Constant term
        
        # For monotonicity, the derivative should not change sign in [0,1]
        
        # If A ≈ 0, derivative is linear
        if abs(A) < epsilon:
            # Linear derivative: B*u + C
            # Check signs at endpoints
            deriv_0 = C
            deriv_1 = B + C
            
            # Monotonic if both endpoints have same sign (or zero)
            same_sign = (deriv_0 >= -epsilon and deriv_1 >= -epsilon) or (deriv_0 <= epsilon and deriv_1 <= epsilon)
            return same_sign
        
        # Quadratic derivative: A*u² + B*u + C
        # Critical point at u_crit = -B / (2*A)
        u_crit = -B / (2 * A)
        
        # Case 1: Critical point is outside [0,1] - automatically monotonic
        if u_crit < -epsilon or u_crit > 1 + epsilon:
            return True
        
        # Case 2: Critical point is inside [0,1] - check if derivative stays same sign
        # Evaluate derivative at critical point and endpoints
        deriv_0 = C
        deriv_1 = A + B + C
        deriv_crit = A * u_crit * u_crit + B * u_crit + C
        
        # Check if all three values have the same sign
        # For increasing monotonic: all should be ≥ 0
        # For decreasing monotonic: all should be ≤ 0
        
        all_non_negative = (deriv_0 >= -epsilon) and (deriv_1 >= -epsilon) and (deriv_crit >= -epsilon)
        all_non_positive = (deriv_0 <= epsilon) and (deriv_1 <= epsilon) and (deriv_crit <= epsilon)
        
        return all_non_negative or all_non_positive