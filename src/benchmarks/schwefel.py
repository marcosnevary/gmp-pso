import math

import jax.numpy as jnp
import numpy as np


def schwefel_fn(x: list | np.ndarray | jnp.ndarray) -> float:
    if isinstance(x, jnp.ndarray):
        n = x.size
        return 418.9829 * n - jnp.sum(x * jnp.sin(jnp.sqrt(jnp.abs(x))))

    if isinstance(x, np.ndarray):
        n = x.size
        return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))

    n = len(x)
    return 418.9829 * n - sum(xi * math.sin(math.sqrt(abs(xi))) for xi in x)

SCHWEFEL_BOUNDS = (-500, 500)
