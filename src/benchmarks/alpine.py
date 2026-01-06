import math

import jax.numpy as jnp
import numpy as np


def alpine_fn(x: list | np.ndarray | jnp.ndarray) -> float:
    if isinstance(x, jnp.ndarray):
        return jnp.sum(jnp.abs(x * jnp.sin(x) + 0.1 * x))

    if isinstance(x, np.ndarray):
        return np.sum(np.abs(x * np.sin(x) + 0.1 * x))

    return sum(abs(xi * math.sin(xi) + 0.1 * xi) for xi in x)

ALPINE_BOUNDS = (-10, 10)
