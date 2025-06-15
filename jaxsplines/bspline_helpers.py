import jax
import jax.numpy as jnp
from jax import vmap
import equinox as eqx
import optax
from typing import Optional, Union, Tuple
from jaxtyping import Array, Float
from .bspline import BSpline


def fit_bspline_to_data(
    target_points: jnp.ndarray,
    n_control_points: int = 10,
    degree: int = 3,
    learning_rate: float = 0.01,
    n_epochs: int = 1000,
    key: jax.random.PRNGKey = jax.random.PRNGKey(42)
):
    """
    Fit a B-spline to target data points using gradient descent.
    
    Args:
        target_points: Target points to fit (shape: [n_points, dim])
        n_control_points: Number of control points for the B-spline
        degree: Polynomial degree
        learning_rate: Learning rate for optimization
        n_epochs: Number of training epochs
        key: Random key for initialization
        
    Returns:
        Trained B-spline and loss history
    """
    # Initialize B-spline with random control points
    spline = create_random_bspline(
        n_control_points=n_control_points,
        dimension=target_points.shape[1],
        degree=degree,
        key=key
    )
    
    # Parameter values for target points (uniform spacing)
    n_target_points = target_points.shape[0]
    target_t = jnp.linspace(0, 1, n_target_points)
    
    # Loss function
    def loss_fn(spline, target_points, target_t):
        predicted_points = spline(target_t)
        return jnp.mean(jnp.sum((predicted_points - target_points) ** 2, axis=1))
    
    # Optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(spline, eqx.is_array))
    
    # Training loop
    loss_history = []
    
    @eqx.filter_jit
    def update_step(spline, opt_state, target_points, target_t):
        loss, grads = eqx.filter_value_and_grad(loss_fn, allow_int=True)(spline, target_points, target_t)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(spline, eqx.is_array))
        spline = eqx.apply_updates(spline, updates)
        return spline, opt_state, loss
    
    for epoch in range(n_epochs):
        spline, opt_state, loss = update_step(spline, opt_state, target_points, target_t)
        loss_history.append(float(loss))
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")
    
    return spline, jnp.array(loss_history)


def create_random_bspline(
    n_control_points: int,
    dimension: int = 2,
    degree: int = 3,
    key: jax.random.PRNGKey = jax.random.PRNGKey(42)
) -> BSpline:
    """
    Create a B-spline with random control points.
    
    Args:
        n_control_points: Number of control points
        dimension: Dimension of the space (2D, 3D, etc.)
        degree: Polynomial degree (1=linear, 2=quadratic, 3=cubic)
        key: Random key for initialization
        
    Returns:
        BSpline with random control points
        
    Note:
        - Only degrees 1-3 are supported
        - Minimum n_control_points should be degree + 1
        - Uses uniform knot spacing
    """
    # Validate degree
    if degree < 1 or degree > 3:
        raise ValueError(f"Only degrees 1-3 are supported, got degree {degree}")
    
    # Ensure we have enough control points for the given degree
    min_control_points = degree + 1
    if n_control_points < min_control_points:
        raise ValueError(
            f"Need at least {min_control_points} control points for degree {degree}, "
            f"got {n_control_points}"
        )
    
    control_points = jax.random.normal(key, (n_control_points, dimension))
    
    return BSpline(
        control_points=control_points,
        degree=degree
    )


def train_step_with_projection(
    spline: BSpline,
    loss_fn: callable,
    optimizer_state: any,
    optimizer: any,
) -> Tuple[BSpline, any, float]:
    """
    Training step with projection to enforce monotonicity constraints.
    
    Args:
        spline: Current B-spline
        loss_fn: Loss function (should take spline as input)
        optimizer_state: Current optimizer state
        optimizer: Optax optimizer
        
    Returns:
        Tuple of (updated spline, new optimizer state, loss value)
    """
    
    # Compute gradients
    loss_value, grads = jax.value_and_grad(loss_fn)(spline)
    
    # Update parameters
    updates, optimizer_state = optimizer.update(grads, optimizer_state, spline)
    updated_spline = eqx.apply_updates(spline, updates)
    
    # Project to satisfy monotonicity constraints (if the method exists)
    if hasattr(updated_spline, 'project_to_monotonic'):
        updated_spline = updated_spline.project_to_monotonic()
    
    return updated_spline, optimizer_state, loss_value