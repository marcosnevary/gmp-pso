import jax.numpy as jnp
import numpy as np


def rosenbrock_fn(x: list | np.ndarray | jnp.ndarray) -> float:
    if isinstance(x, jnp.ndarray):
        return jnp.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)

    if isinstance(x, np.ndarray):
        x = np.asarray(x)
        return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)

    total = 0.0
    for i in range(len(x) - 1):
        total += 100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2
    return total

ROSENBROCK_BOUNDS = (-2.048, 2.048)
