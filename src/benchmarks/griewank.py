import math

import jax.numpy as jnp
import numpy as np


def griewank_fn(x: list | np.ndarray | jnp.ndarray) -> float:
    if isinstance(x, jnp.ndarray):
        i = jnp.arange(1, x.size + 1)
        return 1 + jnp.sum(x**2) / 4000 - jnp.prod(jnp.cos(x / jnp.sqrt(i)))

    if isinstance(x, np.ndarray):
        i = np.arange(1, x.size + 1)
        return 1 + np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(i)))

    prod_cos = 1.0
    for i, xi in enumerate(x, start=1):
        prod_cos *= math.cos(xi / math.sqrt(i))

    return 1 + sum(xi**2 for xi in x) / 4000.0 - prod_cos

GRIEWANK_BOUNDS = (-600, 600)
