import numpy as np


def griewank(x: list | np.ndarray) -> float:
    x = np.asarray(x)
    i = np.arange(1, x.size + 1)
    return 1 + np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(i)))


def rastrigin(x: list | np.ndarray) -> float:
    x = np.asarray(x)
    n = x.size
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def rosenbrock(x: list | np.ndarray) -> float:
    x = np.asarray(x)
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)


def sphere(x: list | np.ndarray) -> float:
    x = np.asarray(x)
    return np.sum(x**2)


benchmarks = {
    'griewank': (griewank, (-5.12, 5.12)),
    'rastrigin': (rastrigin, (-5.12, 5.12)),
    'rosenbrock': (rosenbrock, (-2.048, 2.048)),
    'sphere': (sphere, (-600, 600)),
}
