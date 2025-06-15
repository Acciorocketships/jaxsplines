"""
JAXSplines: B-spline implementation in JAX with Equinox

A JAX-based implementation of B-splines with automatic differentiation support,
designed for machine learning applications.

Key Features:
- JAX-native implementation with full autodiff support
- Equinox-based modular design
- Support for 1D, 2D, and 3D B-splines
- Gradient-based optimization of control points
- vmap compatibility for batch operations
- Monotonicity projection utilities
"""

from .bspline import BSpline
from .bspline_helpers import (
    create_random_bspline,
    fit_bspline_to_data,
    train_step_with_projection
)

__version__ = "0.1.0"
__author__ = "JAXSplines Team"

__all__ = [
    "BSpline",
    "create_random_bspline",
    "fit_bspline_to_data", 
    "train_step_with_projection"
] 