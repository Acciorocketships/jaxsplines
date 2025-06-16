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
            point_0 = self._evaluate_clamped(0.0)
            deriv_0 = self._derivative_clamped(0.0, order=1)
            extrap_neg = point_0 + t_val * deriv_0
            
            # Linear extrapolation to +∞ using derivative at t=1  
            point_1 = self._evaluate_clamped(1.0)
            deriv_1 = self._derivative_clamped(1.0, order=1)
            extrap_pos = point_1 + (t_val - 1.0) * deriv_1
            
            # Normal evaluation within [0,1]
            normal_eval = self._evaluate_clamped(t_val)
            
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
    
    def _evaluate_clamped(self, t_val: float) -> jnp.ndarray:
        """Evaluate B-spline within [0,1] domain (internal helper)."""
        # Get basis function values
        basis_values = self._compute_basis_functions(t_val)
        
        # Compute weighted sum of control points
        return jnp.sum(basis_values[:, None] * self.control_points, axis=0)
    
    def _evaluate_clamped_batch(self, t_vals: jnp.ndarray) -> jnp.ndarray:
        """Evaluate B-spline at multiple points within [0,1] domain (internal helper)."""
        return vmap(self._evaluate_clamped)(t_vals)
    
    def _derivative_clamped(self, t_val: float, order: int = 1) -> jnp.ndarray:
        """Compute derivative within [0,1] domain (internal helper)."""
        if order <= 0:
            return self._evaluate_clamped(t_val)
        
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
        
        # Create derivative spline and evaluate (clamped)
        derivative_spline = BSpline(
            control_points=derivative_control_points,
            degree=current_degree
        )
        
        return derivative_spline._evaluate_clamped(t_val)
    
    def derivative(self, t: Union[float, Float[Array, "..."]], order: int = 1) -> Float[Array, "... dim"]:  # type: ignore
        """
        Compute the derivative of the B-spline at parameter value(s) t.
        For extrapolated regions (t < 0 or t > 1), returns the derivative at the boundary.
        
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
            deriv_0 = self._derivative_clamped(0.0, order)
            deriv_1 = self._derivative_clamped(1.0, order)
            deriv_normal = self._derivative_clamped(t_val, order)
            
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
    
    def invert(self, target_value: float, dimension: int = 0, tolerance: float = 1e-12, 
               assume_monotonic: bool = True) -> float:
        """
        Find the parameter t that gives the target_value in the specified dimension.
        Uses analytical inversion via Cardano's formula for monotonic cubic B-splines.
        
        Args:
            target_value: The desired output value to find t for
            dimension: Which dimension to invert (0 for x, 1 for y, etc.)
            tolerance: Numerical tolerance for degenerate cases
            assume_monotonic: If True, skip monotonicity validation (use after project_to_monotonic)
            
        Returns:
            Parameter value t such that spline(t)[dimension] ≈ target_value
            
        Raises:
            ValueError: If spline is not cubic or fails monotonicity check
        """
        if self.degree != 3:
            raise ValueError("Analytical inversion only implemented for cubic B-splines")
        
        # Rigorous monotonicity check unless explicitly disabled
        if not assume_monotonic:
            if not self.check_monotonic(dimension, tolerance):
                raise ValueError(f"Spline is not monotonic in dimension {dimension}. "
                               f"Use project_to_monotonic() first, or set assume_monotonic=True if you're sure.")
        
        # For monotonic splines, check if we need extrapolation
        point_0 = self._evaluate_clamped(0.0)[dimension]
        point_1 = self._evaluate_clamped(1.0)[dimension] 
        
        # Handle extrapolation cases
        min_boundary = jnp.minimum(point_0, point_1)
        max_boundary = jnp.maximum(point_0, point_1)
        
        if target_value < min_boundary:
            # Extrapolate to negative t values
            deriv_0 = self._derivative_clamped(0.0, order=1)[dimension]
            if abs(deriv_0) < tolerance:
                return 0.0  # Derivative too small, return boundary
            t_extrap = (target_value - point_0) / deriv_0
            return t_extrap
        elif target_value > max_boundary:
            # Extrapolate to t > 1 values  
            deriv_1 = self._derivative_clamped(1.0, order=1)[dimension]
            if abs(deriv_1) < tolerance:
                return 1.0  # Derivative too small, return boundary
            t_extrap = 1.0 + (target_value - point_1) / deriv_1
            return t_extrap
        
        # Find which span contains the target value
        n_control_points = self.control_points.shape[0]
        n_spans = n_control_points - 3
        
        # Evaluate spline at span boundaries to find the right span
        span_params = jnp.linspace(0, 1, n_spans + 1)
        span_values = self(span_params)[:, dimension]
        
        # Find the span where target_value lies
        span_idx = jnp.searchsorted(span_values[:-1], target_value, side='right') - 1
        span_idx = jnp.clip(span_idx, 0, n_spans - 1)
        
        # Get the four control points for this span
        control_points_span = self.control_points[span_idx:span_idx + 4, dimension]
        
        # Convert to local parameter and invert analytically
        u_local = self._invert_span_analytic(target_value, control_points_span, tolerance)
        
        # Convert back to global parameter
        t_global = (span_idx + u_local) / n_spans
        
        return jnp.clip(t_global, 0.0, 1.0)
    
    def check_monotonic(self, dimension: int = 0, epsilon: float = 1e-12) -> bool:
        """
        Check if the spline is monotonic in the specified dimension using the same
        rigorous mathematical conditions as project_to_monotonic.
        
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
            
            # Check discriminant condition for monotonicity:
            # (P₀ - 2P₁ + P₂)² < (P₂ - P₀)(-P₀ + 3P₁ - 3P₂ + P₃)
            lhs = (P0 - 2*P1 + P2)**2
            rhs = (P2 - P0) * (-P0 + 3*P1 - 3*P2 + P3)
            
            if lhs >= rhs + epsilon:  # Fails discriminant condition
                return False
            
            # Check vertex-outside conditions
            if not self._check_vertex_outside_conditions(P0, P1, P2, P3, epsilon):
                return False
        
        return True
    
    def _check_vertex_outside_conditions(self, P0: float, P1: float, P2: float, P3: float, epsilon: float) -> bool:
        """
        Check if the vertex of the cubic's derivative is outside [0,1] interval,
        using the same conditions as the projection methods.
        """
        # The derivative of the cubic B-spline has a critical point.
        # For monotonicity, this critical point should be outside [0,1].
        
        # Coefficients of the derivative (quadratic)
        # Derivative: A'u² + B'u + C' where:
        A_deriv = (-P0 + 3*P1 - 3*P2 + P3) / 2  # Coefficient of u²
        B_deriv = (P0 - 2*P1 + P2)              # Coefficient of u
        C_deriv = (P2 - P0) / 2                 # Constant term
        
        # For monotonicity, either:
        # 1. No critical point (A_deriv ≈ 0), or
        # 2. Critical point outside [0,1]
        
        if abs(A_deriv) < epsilon:
            # Linear derivative - always monotonic if slope doesn't change sign
            return True
        
        # Critical point at u_crit = -B_deriv / (2 * A_deriv)
        u_crit = -B_deriv / (2 * A_deriv)
        
        # Check if critical point is outside [0,1]
        outside_interval = (u_crit < -epsilon) or (u_crit > 1 + epsilon)
        
        if outside_interval:
            return True
        
        # Alternative conditions from projection methods:
        # Left of 0 case: P₀ ≥ 2P₁ - P₂
        left_condition = P0 >= 2*P1 - P2 - epsilon
        
        # Right of 1 case: P₂ ≥ (P₃ + P₁) / 2  
        right_condition = P2 >= (P3 + P1) / 2 - epsilon
        
        # Common condition: P₃ > P₀ - 3P₁ + 3P₂
        common_condition = P3 > P0 - 3*P1 + 3*P2 - epsilon
        
        # Monotonic if any of the vertex-outside conditions are satisfied
        return common_condition and (left_condition or right_condition)
    
    def _invert_span_analytic(self, y: float, P: jnp.ndarray, tolerance: float = 1e-12) -> float:
        """
        Analytically invert one cubic B-spline span using Cardano's formula.
        
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
        
        # Handle degenerate cases
        if jnp.abs(A) < tolerance:
            # Quadratic or linear case
            if jnp.abs(B) < tolerance:
                # Linear case: C*u + F = 0
                if jnp.abs(C) < tolerance:
                    # Constant case - should not happen with proper input
                    raise ValueError("Constant case - should not happen with proper input")
                else:
                    u = -F / C
                    return jnp.clip(u, 0.0, 1.0)
            else:
                # Quadratic case: B*u² + C*u + F = 0
                discriminant = C*C - 4*B*F
                if discriminant < 0:
                    raise ValueError("No real solution")
                sqrt_disc = jnp.sqrt(discriminant)
                u1 = (-C + sqrt_disc) / (2*B)
                u2 = (-C - sqrt_disc) / (2*B)
                
                # Choose root in [0,1]
                if 0 <= u1 <= 1:
                    return u1
                elif 0 <= u2 <= 1:
                    return u2
                else:
                    return jnp.clip(u1, 0.0, 1.0)
        
        # Full cubic case - depress the cubic
        # Transform to v³ + p*v + q = 0 with u = v - B/(3*A)
        p = (3*A*C - B*B) / (3*A*A)
        q = (2*B*B*B - 9*A*B*C + 27*A*A*F) / (27*A*A*A)
        
        # Cardano's formula
        Delta = (q/2)**2 + (p/3)**3
        
        if Delta >= 0:
            # One real root
            sqrt_delta = jnp.sqrt(Delta)
            
            # Handle negative cube roots properly
            term1 = -q/2 + sqrt_delta
            term2 = -q/2 - sqrt_delta
            
            cbrt1 = jnp.sign(term1) * jnp.abs(term1)**(1/3)
            cbrt2 = jnp.sign(term2) * jnp.abs(term2)**(1/3)
            
            v = cbrt1 + cbrt2
            u = v - B/(3*A)
            
            return jnp.clip(u, 0.0, 1.0)
        
        else:
            # Three real roots
            r = 2 * jnp.sqrt(-p/3)
            phi = jnp.arccos(-q/2 / jnp.sqrt(-(p/3)**3))
            
            # Compute all three roots
            u0 = r * jnp.cos(phi/3) - B/(3*A)
            u1 = r * jnp.cos((phi + 2*jnp.pi)/3) - B/(3*A)
            u2 = r * jnp.cos((phi + 4*jnp.pi)/3) - B/(3*A)
            
            # Choose the root that lies in [0,1]
            roots = jnp.array([u0, u1, u2])
            valid_mask = (roots >= 0) & (roots <= 1)
            
            # If we have valid roots, choose the first one
            # If no valid roots, clamp the closest one
            if jnp.any(valid_mask):
                valid_roots = jnp.where(valid_mask, roots, jnp.inf)
                return jnp.min(valid_roots)
            else:
                # Choose the root closest to [0,1]
                distances = jnp.minimum(jnp.abs(roots), jnp.abs(roots - 1))
                closest_idx = jnp.argmin(distances)
                return jnp.clip(roots[closest_idx], 0.0, 1.0)
    
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