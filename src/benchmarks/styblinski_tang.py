import jax.numpy as jnp
import numpy as np


def styblinski_tang_fn(x: list | np.ndarray | jnp.ndarray) -> float:
    if isinstance(x, jnp.ndarray):
        return 0.5 * jnp.sum(x**4 - 16 * x**2 + 5 * x)

    if isinstance(x, np.ndarray):
        return 0.5 * np.sum(x**4 - 16 * x**2 + 5 * x)

    return 0.5 * sum(xi**4 - 16 * xi**2 + 5 * xi for xi in x)

STYBLINSKI_TANG_BOUNDS = (-5, 5)
