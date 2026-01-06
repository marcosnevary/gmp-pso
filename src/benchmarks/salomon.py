import math

import jax.numpy as jnp
import numpy as np


def salomon_fn(x: list | np.ndarray | jnp.ndarray) -> float:
    if isinstance(x, jnp.ndarray):
        r = jnp.sqrt(jnp.sum(x**2))
        return 1 - jnp.cos(2 * jnp.pi * r) + 0.1 * r

    if isinstance(x, np.ndarray):
        r = np.sqrt(np.sum(x**2))
        return 1 - np.cos(2 * np.pi * r) + 0.1 * r

    r = math.sqrt(sum(xi**2 for xi in x))
    return 1 - math.cos(2 * math.pi * r) + 0.1 * r

SALOMON_BOUNDS = (-100, 100)
