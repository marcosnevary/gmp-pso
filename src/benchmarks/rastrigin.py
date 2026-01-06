import math

import jax.numpy as jnp
import numpy as np


def rastrigin_fn(x: list | np.ndarray | jnp.ndarray) -> float:
    if isinstance(x, jnp.ndarray):
        n = x.size
        return 10 * n + jnp.sum(x**2 - 10 * jnp.cos(2 * jnp.pi * x))

    if isinstance(x, np.ndarray):
        n = x.size
        return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

    n = len(x)
    return 10 * n + sum(xi**2 - 10 * math.cos(2 * math.pi * xi) for xi in x)

RASTRIGIN_BOUNDS = (-5.12, 5.12)
