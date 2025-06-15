# B-splines in JAX with Equinox

A comprehensive, high-performance implementation of B-splines in JAX with Equinox, designed for machine learning applications with learnable parameters and efficient parallelization.

## Features

- 🚀 **JAX-native**: Fully compatible with JAX transformations (jit, vmap, grad)
- 🎯 **Learnable Parameters**: Control points and optionally knot vectors as learnable parameters
- ⚡ **Vectorized Operations**: Efficient evaluation using vmap for parallel processing
- 🔧 **Equinox Integration**: Clean PyTree structure for seamless ML integration
- 📐 **Multiple Dimensions**: Support for 2D, 3D, and higher-dimensional splines
- 🎨 **Degrees 1-3**: Support for linear, quadratic, and cubic B-splines
- 📊 **Derivatives**: Analytical computation of derivatives up to arbitrary order
- 🔄 **Optimization Ready**: Direct integration with JAX optimizers (Optax)

## Installation

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/username/jaxsplines.git
cd jaxsplines

# Install in development mode
pip install -e .

# Or install with optional dependencies
pip install -e ".[dev,examples]"
```

### From PyPI (Future)

```bash
pip install jaxsplines
```

### Dependencies

- JAX >= 0.4.20
- Equinox >= 0.11.0
- Optax >= 0.1.7
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0 (for visualization examples)
- jaxtyping >= 0.2.20

## Quick Start

### Basic Usage

```python
import jax.numpy as jnp
from jaxsplines import BSpline

# Define control points
control_points = jnp.array([
    [0.0, 0.0],
    [1.0, 2.0],
    [2.0, -1.0],
    [3.0, 1.0],
    [4.0, 0.0]
])

# Create B-spline
spline = BSpline(control_points=control_points, degree=3)

# Evaluate spline
t_values = jnp.linspace(0, 1, 100)
points = spline(t_values)  # Shape: (100, 2)

print(f"Evaluated spline at {len(t_values)} points")
print(f"Output shape: {points.shape}")
```

### Vectorized Evaluation with vmap

```python
import jax

# Batch evaluation - evaluate spline at multiple parameter sets
t_batch = jnp.array([
    [0.0, 0.5, 1.0],  # First parameter set
    [0.2, 0.6, 0.8]   # Second parameter set
])

# Use vmap for parallel evaluation
points_batch = jax.vmap(spline)(t_batch)
print(f"Batch evaluation shape: {points_batch.shape}")  # (2, 3, 2)
```

### Training with Learnable Parameters

```python
import jax
import optax
import equinox as eqx
from jaxsplines import create_random_bspline

# Create target data (e.g., sine wave)
t_target = jnp.linspace(0, 2 * jnp.pi, 50)
target_points = jnp.stack([t_target, jnp.sin(t_target)], axis=1)

# Initialize learnable B-spline
spline = create_random_bspline(
    n_control_points=10,
    dimension=2,
    degree=3,
    key=jax.random.PRNGKey(42)
)

# Define loss function
def loss_fn(spline, target_points):
    t_eval = jnp.linspace(0, 1, len(target_points))
    predicted_points = spline(t_eval)
    return jnp.mean(jnp.sum((predicted_points - target_points) ** 2, axis=1))

# Setup optimizer
optimizer = optax.adam(0.01)
opt_state = optimizer.init(eqx.filter(spline, eqx.is_array))

# Training step
@eqx.filter_jit
def update_step(spline, opt_state, target_points):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(spline, target_points)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(spline, eqx.is_array))
    spline = eqx.apply_updates(spline, updates)
    return spline, opt_state, loss

# Training loop
for epoch in range(1000):
    spline, opt_state, loss = update_step(spline, opt_state, target_points)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")
```

## API Reference

### BSpline Class

The main B-spline class with learnable parameters.

```python
class BSpline(eqx.Module):
    def __init__(
        self,
        control_points: Float[Array, "n_control_points dim"],
        knots: Optional[Float[Array, "n_knots"]] = None,
        degree: int = 3,
        learnable_knots: bool = False,
        key: Optional[jax.random.PRNGKey] = None
    )
```

**Parameters:**
- `control_points`: Control points defining the spline shape
- `knots`: Knot vector (auto-generated if None)
- `degree`: Polynomial degree of the B-spline
- `learnable_knots`: Whether knot positions should be learnable
- `key`: Random key for initialization

**Methods:**

#### `__call__(t)`
Evaluate the B-spline at parameter value(s) t.

```python
# Single point
point = spline(0.5)  # Shape: (dim,)

# Multiple points
points = spline(jnp.array([0.0, 0.5, 1.0]))  # Shape: (3, dim)

# Batch evaluation
t_batch = jnp.array([[0.0, 0.5], [0.3, 0.8]])
points_batch = jax.vmap(spline)(t_batch)  # Shape: (2, 2, dim)
```

#### `derivative(t, order=1)`
Compute derivatives of the B-spline.

```python
# First derivative (tangent vectors)
tangents = spline.derivative(t_values, order=1)

# Second derivative (acceleration)
accelerations = spline.derivative(t_values, order=2)
```

#### `arc_length(t_start=0.0, t_end=1.0, n_samples=1000)`
Compute arc length using numerical integration.

```python
length = spline.arc_length(0.0, 1.0)
```

### Utility Functions

#### `create_random_bspline()`
Create a B-spline with random control points.

```python
spline = create_random_bspline(
    n_control_points=8,
    dimension=2,
    degree=3,
    key=jax.random.PRNGKey(42)
)
```

#### `create_circle_bspline()`
Create a B-spline approximating a circle.

```python
circle_spline = create_circle_bspline(
    radius=2.0,
    center=jnp.array([1.0, 1.0]),
    n_control_points=8
)
```

## Advanced Usage

### 3D Splines

```python
# 3D control points
control_points_3d = jax.random.normal(jax.random.PRNGKey(0), (8, 3))
spline_3d = BSpline(control_points=control_points_3d, degree=3)

# Evaluate in 3D
t_values = jnp.linspace(0, 1, 100)
points_3d = spline_3d(t_values)  # Shape: (100, 3)
```

### Custom Knot Vectors

```python
# Define custom knot vector
knots = jnp.array([0.0, 0.0, 0.0, 0.0, 0.3, 0.7, 1.0, 1.0, 1.0, 1.0])
spline = BSpline(
    control_points=control_points,
    knots=knots,
    degree=3
)
```

### Learnable Knots

```python
# Enable learnable knot positions
spline = BSpline(
    control_points=control_points,
    degree=3,
    learnable_knots=True  # Knots become learnable parameters
)
```

### Batch Processing Multiple Splines

```python
# Create batch of control points
batch_size = 5
control_points_batch = jax.random.normal(
    jax.random.PRNGKey(0), 
    (batch_size, 8, 2)
)

# Create batch of splines
def create_spline(control_points):
    return BSpline(control_points=control_points, degree=3)

splines_batch = jax.vmap(create_spline)(control_points_batch)

# Batch evaluation
t_values = jnp.linspace(0, 1, 50)
def evaluate_spline(spline):
    return spline(t_values)

results = jax.vmap(evaluate_spline)(splines_batch)
print(f"Batch results shape: {results.shape}")  # (5, 50, 2)
```

## Performance Considerations

### JIT Compilation

For optimal performance, use JIT compilation:

```python
@jax.jit
def evaluate_spline(spline, t_values):
    return spline(t_values)

# First call compiles the function
points = evaluate_spline(spline, t_values)

# Subsequent calls are fast
points = evaluate_spline(spline, t_values)  # Fast!
```

### Memory Efficiency

For large-scale applications:

```python
# Use chunked evaluation for very large t_values arrays
def chunked_evaluation(spline, t_values, chunk_size=1000):
    n_chunks = len(t_values) // chunk_size + 1
    results = []
    
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(t_values))
        chunk = t_values[start_idx:end_idx]
        
        if len(chunk) > 0:
            results.append(spline(chunk))
    
    return jnp.concatenate(results, axis=0)
```

## Examples

Run the example scripts to see the implementation in action:

```bash
# Basic usage and training examples
python examples/example.py

# Run comprehensive tests
python -m pytest tests/

# Or run tests directly
python tests/test_bspline.py
```

The example script demonstrates:
- Fitting B-splines to various curves (sine wave, circle, 3D helix)
- Training with gradient descent
- Visualization of results
- Performance benchmarking
- vmap parallelization

## Mathematical Background

B-splines are defined by:
- **Control points**: Points that influence the curve shape
- **Knot vector**: Determines parameter ranges for basis functions
- **Degree**: Polynomial degree of the basis functions

The curve is computed as:
```
C(t) = Σᵢ Nᵢ,ₚ(t) × Pᵢ
```

Where:
- `Nᵢ,ₚ(t)` are B-spline basis functions of degree p
- `Pᵢ` are control points
- Basis functions use optimized direct formulas for degrees 1-3

### Properties

- **Local support**: Changes to control points only affect nearby curve regions
- **Convex hull property**: Curve lies within convex hull of control points
- **Partition of unity**: Basis functions sum to 1 at any parameter value
- **Smoothness**: Curve is Cᵖ⁻¹ continuous (p = degree)

## Contributing

Contributions are welcome! Please ensure:
1. Code follows JAX best practices
2. All tests pass (`python test_bspline.py`)
3. New features include appropriate tests
4. Documentation is updated

## License

MIT License - see LICENSE file for details.

## Citation

If you use this implementation in research, please cite:

```bibtex
@software{jax_bsplines,
  title={B-splines in JAX with Equinox},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/jax-bsplines}
}
``` 