from functools import partial
from typing import NamedTuple

import jax.numpy as jnp
from jax import grad, jit, lax, random, vmap


class SwarmState(NamedTuple):
    positions: jnp.ndarray
    velocities: jnp.ndarray
    p_best_pos: jnp.ndarray
    p_best_fit: jnp.ndarray
    g_best_pos: jnp.ndarray
    g_best_fit: jnp.ndarray
    rng: random.PRNGKey

class GradientState(NamedTuple):
    new_g_best_pos: jnp.ndarray
    lower: jnp.ndarray
    upper: jnp.ndarray
    eta: float

@partial(
    jit,
    static_argnames=(
        'objective_fn',
        'bounds',
        'num_dims',
        'num_particles',
        'max_iters',
        'max_epochs',
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
    eta: float,
    max_epochs: int,
) -> tuple:
    lower, upper = bounds
    k_pos, k_vel, k_state = random.split(key, 3)

    search_range = upper - lower
    velocity_scale = 0.1
    limit = search_range * velocity_scale

    init_positions = random.uniform(
        k_pos, (num_particles, num_dims), minval=lower, maxval=upper,
    )
    init_velocities = random.uniform(
        k_vel, (num_particles, num_dims), minval=-limit, maxval=limit,
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

    gradient_fn = grad(objective_fn)

    def update_step(swarm_state: SwarmState, i: int) -> tuple:
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
        candidate_g_pos = jnp.where(
            global_improved, new_p_best_pos[current_g_best_idx], swarm_state.g_best_pos,
        )
        candidate_g_fit = jnp.where(
            global_improved, current_g_best_fit, swarm_state.g_best_fit,
        )

        def gradient_descent_step(g_state: GradientState, _: None) -> tuple:
            grads = gradient_fn(g_state.new_g_best_pos)
            step = g_state.eta * grads
            updated_pos = g_state.new_g_best_pos - step
            updated_pos = jnp.clip(updated_pos, g_state.lower, g_state.upper)
            next_state = GradientState(
                new_g_best_pos=updated_pos,
                lower=g_state.lower,
                upper=g_state.upper,
                eta=g_state.eta,
            )
            return next_state, None

        def apply_gradient(_: None) -> tuple:
            init_grad_state = GradientState(candidate_g_pos, lower, upper, eta)
            final_grad_state, _ = lax.scan(
                gradient_descent_step,
                init_grad_state,
                None,
                max_epochs,
            )
            final_pos = final_grad_state.new_g_best_pos
            return final_pos, objective_fn(final_pos)

        def skip_gradient(_: None) -> tuple:
            return candidate_g_pos, candidate_g_fit

        gradient_g_pos, gradient_g_fit = lax.cond(
            i % 10 == 0,
            apply_gradient,
            skip_gradient,
            None,
        )

        global_improved = gradient_g_fit < candidate_g_fit
        final_g_pos = jnp.where(global_improved, gradient_g_pos, candidate_g_pos)
        final_g_fit = jnp.where(global_improved, gradient_g_fit, candidate_g_fit)

        next_state = SwarmState(
            positions=new_positions,
            velocities=new_velocities,
            p_best_pos=new_p_best_pos,
            p_best_fit=new_p_best_fit,
            g_best_pos=final_g_pos,
            g_best_fit=final_g_fit,
            rng=k_next,
        )

        return next_state, final_g_fit

    final_state, history = lax.scan(update_step, initial_state, jnp.arange(max_iters))
    full_history = jnp.concatenate([jnp.array([initial_state.g_best_fit]), history])

    return final_state.g_best_pos, final_state.g_best_fit, full_history
