import jax.numpy as jnp
import numpy as np


def sphere_fn(x: list | np.ndarray | jnp.ndarray) -> float:
    if isinstance(x, jnp.ndarray):
        return jnp.sum(x**2)

    if isinstance(x, np.ndarray):
        return np.sum(x**2)

    return sum(xi**2 for xi in x)

SPHERE_BOUNDS = (-5.12, 5.12)
