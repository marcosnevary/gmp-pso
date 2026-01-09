from functools import partial
from typing import NamedTuple

import jax.numpy as jnp
from jax import jit, lax, random, vmap


class SwarmState(NamedTuple):
    positions: jnp.ndarray
    velocities: jnp.ndarray
    p_best_pos: jnp.ndarray
    p_best_fit: jnp.ndarray
    g_best_pos: jnp.ndarray
    g_best_fit: jnp.ndarray
    rng: random.PRNGKey

@partial(
    jit,
    static_argnames=(
        'objective_fn',
        'bounds',
        'num_dims',
        'num_particles',
        'max_iters',
        'c1',
        'c2',
        'w',
    ),
)
def jax_pso(
    objective_fn: callable,
    bounds: tuple,
    num_dims: int,
    num_particles: int,
    max_iters: int,
    c1: float,
    c2: float,
    w: float,
    key: random.PRNGKey,
) -> tuple:
    lower, upper = bounds
    k_pos, k_vel, k_state = random.split(key, 3)

    init_positions = random.uniform(
        k_pos, (num_particles, num_dims), minval=lower, maxval=upper,
    )
    init_velocities = random.uniform(
        k_vel, (num_particles, num_dims), minval=-1.0, maxval=1.0,
    )
    init_fitness = vmap(objective_fn)(init_positions)

    best_idx = jnp.argmin(init_fitness)
    g_best_pos = init_positions[best_idx]
    g_best_fit = init_fitness[best_idx]

    initial_state = SwarmState(
        positions=init_positions,
        velocities=init_velocities,
        p_best_pos=init_positions,
        p_best_fit=init_fitness,
        g_best_pos=g_best_pos,
        g_best_fit=g_best_fit,
        rng=k_state,
    )

    def update_step(swarm_state: SwarmState, _: None) -> tuple:
        k1, k2, k_next = random.split(swarm_state.rng, 3)
        r1 = random.uniform(k1, (num_particles, num_dims))
        r2 = random.uniform(k2, (num_particles, num_dims))

        inertia = w * swarm_state.velocities
        cognitive = c1 * r1 * (swarm_state.p_best_pos - swarm_state.positions)
        social = c2 * r2 * (swarm_state.g_best_pos - swarm_state.positions)

        new_velocities = inertia + cognitive + social
        new_positions = swarm_state.positions + new_velocities
        new_positions = jnp.clip(new_positions, lower, upper)

        new_fitness = vmap(objective_fn)(new_positions)

        improved = new_fitness < swarm_state.p_best_fit
        mask = improved[:, None]
        new_p_best_pos = jnp.where(mask, new_positions, swarm_state.p_best_pos)
        new_p_best_fit = jnp.where(improved, new_fitness, swarm_state.p_best_fit)

        current_g_best_idx = jnp.argmin(new_p_best_fit)
        current_g_best_fit = new_p_best_fit[current_g_best_idx]
        global_improved = current_g_best_fit < swarm_state.g_best_fit
        new_g_best_pos = jnp.where(
            global_improved, new_p_best_pos[current_g_best_idx], swarm_state.g_best_pos,
        )
        new_g_best_fit = jnp.where(
            global_improved, current_g_best_fit, swarm_state.g_best_fit,
        )

        next_state = SwarmState(
            positions=new_positions,
            velocities=new_velocities,
            p_best_pos=new_p_best_pos,
            p_best_fit=new_p_best_fit,
            g_best_pos=new_g_best_pos,
            g_best_fit=new_g_best_fit,
            rng=k_next,
        )

        return next_state, None

    final_state, _ = lax.scan(update_step, initial_state, None, max_iters)
    return final_state.g_best_pos, final_state.g_best_fit
