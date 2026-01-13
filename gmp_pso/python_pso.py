import random
from typing import NamedTuple


class SwarmState(NamedTuple):
    positions: list
    velocities: list
    p_best_pos: list
    p_best_fit: list
    g_best_pos: list
    g_best_fit: float
    rng: random.Random
    history: list

def python_pso(
    objective_fn: callable,
    bounds: tuple,
    num_dims: int,
    num_particles: int,
    max_iters: int,
    c1: float,
    c2: float,
    w: float,
    seed: int,
    **_: any,
) -> tuple:
    lower, upper = bounds
    rng = random.Random(seed)

    init_positions = [
        [rng.uniform(lower, upper) for _ in range(num_dims)]
        for _ in range(num_particles)
    ]
    init_velocities = [
        [rng.uniform(-1.0, 1.0) for _ in range(num_dims)] for _ in range(num_particles)
    ]
    init_fitness = [objective_fn(pos) for pos in init_positions]

    best_idx = init_fitness.index(min(init_fitness))
    g_best_pos = init_positions[best_idx]
    g_best_fit = init_fitness[best_idx]

    history = [g_best_fit]

    swarm_state = SwarmState(
        positions=init_positions,
        velocities=init_velocities,
        p_best_pos=init_positions,
        p_best_fit=init_fitness,
        g_best_pos=g_best_pos,
        g_best_fit=g_best_fit,
        rng=rng,
        history=history,
    )

    for _ in range(max_iters):
        r1 = [
            [swarm_state.rng.random() for _ in range(num_dims)]
            for _ in range(num_particles)
        ]
        r2 = [
            [swarm_state.rng.random() for _ in range(num_dims)]
            for _ in range(num_particles)
        ]
        inertia = [
            [w * v for v in swarm_state.velocities[i]] for i in range(num_particles)
        ]
        cognitive = [
            [
                c1 * r1[i][d] * (
                    swarm_state.p_best_pos[i][d] - swarm_state.positions[i][d]
                )
                for d in range(num_dims)
            ]
            for i in range(num_particles)
        ]
        social = [
            [
                c2 * r2[i][d] * (
                    swarm_state.g_best_pos[d] - swarm_state.positions[i][d]
                )
                for d in range(num_dims)
            ]
            for i in range(num_particles)
        ]

        new_velocities = [
            [
                inertia[i][d] + cognitive[i][d] + social[i][d]
                for d in range(num_dims)
            ]
            for i in range(num_particles)
        ]
        new_positions = [
            [
                max(
                    lower,
                    min(upper, swarm_state.positions[i][d] + new_velocities[i][d]),
                )
                for d in range(num_dims)
            ]
            for i in range(num_particles)
        ]

        current_fitness = [objective_fn(pos) for pos in new_positions]

        new_p_best_pos = list(swarm_state.p_best_pos)
        new_p_best_fit = list(swarm_state.p_best_fit)
        new_g_best_pos = swarm_state.g_best_pos
        new_g_best_fit = swarm_state.g_best_fit

        for i in range(num_particles):
            if current_fitness[i] < swarm_state.p_best_fit[i]:
                new_p_best_pos[i] = new_positions[i][:]
                new_p_best_fit[i] = current_fitness[i]

            if current_fitness[i] < new_g_best_fit:
                new_g_best_pos = new_positions[i][:]
                new_g_best_fit = current_fitness[i]

        new_history = swarm_state.history.append(new_g_best_fit)

        swarm_state = SwarmState(
            positions=new_positions,
            velocities=new_velocities,
            p_best_pos=new_p_best_pos,
            p_best_fit=new_p_best_fit,
            g_best_pos=new_g_best_pos,
            g_best_fit=new_g_best_fit,
            rng=swarm_state.rng,
            history=new_history,
        )

    return swarm_state.g_best_pos, swarm_state.g_best_fit, swarm_state.history
