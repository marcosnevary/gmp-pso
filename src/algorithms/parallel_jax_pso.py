import jax.numpy as jnp
from jax import block_until_ready, random, vmap

from .jax_pso import jax_pso


def parallel_jax_pso(
    objective_fn: callable,
    bounds: tuple,
    params: dict,
    num_subswarms: int,
) -> tuple:
    key = random.PRNGKey(params['seed'])
    keys = random.split(key, num_subswarms)

    parallel_pso = vmap(
        jax_pso,
        in_axes=(None, None, None, None, None, None, None, None, 0),
    )

    all_best_positions, all_best_fitnesses = parallel_pso(
        objective_fn,
        bounds,
        params['num_dims'],
        params['num_particles'],
        params['max_iters'],
        params['c1'],
        params['c2'],
        params['w'],
        keys,
    )

    idx_best_subwarm = jnp.argmin(all_best_fitnesses)

    best_position = all_best_positions[idx_best_subwarm]
    best_fitness = all_best_fitnesses[idx_best_subwarm]

    result = (best_position, best_fitness)

    return block_until_ready(result)
