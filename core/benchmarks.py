import math

import jax.numpy as jnp
import numpy as np
from jax import jit

from .jax_gd_pso import jax_gd_pso
from .numpy_pso import numpy_pso


def ackley_py(x: list) -> float:
    n = len(x)
    sum1 = sum(xi**2 for xi in x)
    sum2 = sum(math.cos(2 * math.pi * xi) for xi in x)
    return -20 * math.exp(-0.2 * math.sqrt(sum1 / n)) - math.exp(sum2 / n) + 20 + math.e


def rastrigin_py(x: list) -> float:
    n = len(x)
    return 10 * n + sum(xi**2 - 10 * math.cos(2 * math.pi * xi) for xi in x)


def sphere_py(x: list) -> float:
    return sum(xi**2 for xi in x)


def rosenbrock_py(x: list) -> float:
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2 for i in range(len(x) - 1))


def ackley_np(x: np.ndarray) -> float:
    n = x.shape[0]
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e


def rastrigin_np(x: np.ndarray) -> float:
    n = x.shape[0]
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def sphere_np(x: np.ndarray) -> float:
    return np.sum(x**2)


def rosenbrock_np(x: np.ndarray) -> float:
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


@jit
def ackley_jax(x: jnp.ndarray) -> jnp.ndarray:
    n = x.shape[0]
    sum1 = jnp.sum(x**2)
    sum2 = jnp.sum(jnp.cos(2 * jnp.pi * x))
    return -20 * jnp.exp(-0.2 * jnp.sqrt(sum1 / n)) - jnp.exp(sum2 / n) + 20 + jnp.e


@jit
def rastrigin_jax(x: jnp.ndarray) -> jnp.ndarray:
    n = x.shape[0]
    return 10 * n + jnp.sum(x**2 - 10 * jnp.cos(2 * jnp.pi * x))


@jit
def sphere_jax(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(x**2)


@jit
def rosenbrock_jax(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


BENCHMARKS = {
    "Ackley": {
        "bounds": (-32.768, 32.768),
        "Python PSO": ackley_py,
        "NumPy PSO": ackley_np,
        "JAX PSO": ackley_jax,
    },
    "Rastrigin": {
        "bounds": (-5.12, 5.12),
        "Python PSO": rastrigin_py,
        "NumPy PSO": rastrigin_np,
        "JAX PSO": rastrigin_jax,
    },
    "Rosenbrock": {
        "bounds": (-5.0, 10.0),
        "Python PSO": rosenbrock_py,
        "NumPy PSO": rosenbrock_np,
        "JAX PSO": rosenbrock_jax,
    },
    "Sphere": {
        "bounds": (-5.12, 5.12),
        "Python PSO": sphere_py,
        "NumPy PSO": sphere_np,
        "JAX PSO": sphere_jax,
    },
}

ALGORITHMS = {
    "NumPy PSO": numpy_pso,
    "JAX PSO": jax_gd_pso,
}

DIMS = [10, 30, 50, 100]

HYPERPARAMETERS = {
    "num_dims": None,
    "num_particles": 30,
    "max_iters": 100,
    "c1": 1.5,
    "c2": 1.5,
    "w": 0.7,
    "seed": None,
    "eta": 0.001,
    "steps": 5,
}

NUM_RUNS = 10
