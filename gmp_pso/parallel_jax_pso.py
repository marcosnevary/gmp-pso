import jax.numpy as jnp
from jax import block_until_ready, random, vmap

from .jax_pso import jax_pso


def parallel_jax_pso(
    objective_fn: callable,
    bounds: tuple,
    num_dims: int,
    num_particles: int,
    max_iters: int,
    c1: float,
    c2: float,
    w: float,
    seed: int,
    eta: float,
    num_subswarms: int,
) -> tuple:
    key = random.PRNGKey(seed)
    keys = random.split(key, num_subswarms)

    parallel_pso = vmap(
        jax_pso,
        in_axes=(None, None, None, None, None, None, None, None, 0, None),
    )

    all_best_positions, all_best_fitnesses, all_histories = parallel_pso(
        objective_fn,
        bounds,
        num_dims,
        num_particles,
        max_iters,
        c1,
        c2,
        w,
        keys,
        eta,
    )

    idx_best_subwarm = jnp.argmin(all_best_fitnesses)

    best_position = all_best_positions[idx_best_subwarm]
    best_fitness = all_best_fitnesses[idx_best_subwarm]

    result = (best_position, best_fitness, all_histories)

    return block_until_ready(result)
