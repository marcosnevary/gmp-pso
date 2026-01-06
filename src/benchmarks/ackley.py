import math

import jax.numpy as jnp
import numpy as np


def ackley_fn(x: list | np.ndarray | jnp.ndarray) -> float:
    if isinstance(x, jnp.ndarray):
        n = x.size
        sum1 = jnp.sum(x**2)
        sum2 = jnp.sum(jnp.cos(2 * jnp.pi * x))
        return -20 * jnp.exp(-0.2 * jnp.sqrt(sum1 / n)) - jnp.exp(sum2 / n) + 20 + jnp.e

    if isinstance(x, np.ndarray):
        n = x.size
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(2 * np.pi * x))
        return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e

    n = len(x)
    sum1 = sum(xi**2 for xi in x)
    sum2 = sum(math.cos(2 * math.pi * xi) for xi in x)
    return -20 * math.exp(-0.2 * math.sqrt(sum1 / n)) - math.exp(sum2 / n) + 20 + math.e

ACKLEY_BOUNDS = (-32.768, 32.768)
