from typing import NamedTuple

import numpy as np


class SwarmState(NamedTuple):
    positions: np.ndarray
    velocities: np.ndarray
    p_best_pos: np.ndarray
    p_best_fit: np.ndarray
    g_best_pos: np.ndarray
    g_best_fit: np.ndarray
    rng: np.random.Generator

def vectorized_pso(
    objective_fn: callable,
    bounds: tuple,
    num_dims: int,
    num_particles: int,
    max_iters: int,
    c1: float,
    c2: float,
    w: float,
    seed: int,
) -> tuple:
    lower, upper = bounds
    rng = np.random.default_rng(seed)

    init_positions = rng.uniform(lower, upper, (num_particles, num_dims))
    init_velocities = rng.uniform(-1.0, 1.0, (num_particles, num_dims))
    init_fitness = np.array([objective_fn(pos) for pos in init_positions])

    best_idx = np.argmin(init_fitness)
    g_best_pos = init_positions[best_idx]
    g_best_fit = init_fitness[best_idx]

    swarm_state = SwarmState(
        positions=init_positions,
        velocities=init_velocities,
        p_best_pos=init_positions,
        p_best_fit=init_fitness,
        g_best_pos=g_best_pos,
        g_best_fit=g_best_fit,
        rng=rng,
    )

    for _ in range(max_iters):
        r1 = swarm_state.rng.random((num_particles, num_dims))
        r2 = swarm_state.rng.random((num_particles, num_dims))

        inertia = w * swarm_state.velocities
        cognitive = c1 * r1 * (swarm_state.p_best_pos - swarm_state.positions)
        social = c2 * r2 * (swarm_state.g_best_pos - swarm_state.positions)

        new_velocities = inertia + cognitive + social
        new_positions = swarm_state.positions + new_velocities
        new_positions = np.clip(new_positions, lower, upper)

        current_fitness = np.array([objective_fn(pos) for pos in new_positions])

        improved = current_fitness < swarm_state.p_best_fit
        mask = improved[:, None]
        new_p_best_pos = np.where(mask, new_positions, swarm_state.p_best_pos)
        new_p_best_fit = np.where(improved, current_fitness, swarm_state.p_best_fit)

        best_idx = np.argmin(current_fitness)
        new_g_best_pos = new_positions[best_idx].copy()
        new_g_best_fit = current_fitness[best_idx]

        swarm_state = SwarmState(
            positions=new_positions,
            velocities=new_velocities,
            p_best_pos=new_p_best_pos,
            p_best_fit=new_p_best_fit,
            g_best_pos=new_g_best_pos,
            g_best_fit=new_g_best_fit,
            rng=swarm_state.rng,
        )

    return swarm_state.g_best_pos, swarm_state.g_best_fit
