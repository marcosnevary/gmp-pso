import time

import jax.numpy as jnp
import pandas as pd
from jax import block_until_ready, random

from core.benchmarks import ALGORITHMS, BENCHMARKS, DIMS, HYPERPARAMETERS, NUM_RUNS
from core.plot_benchmarks import generate_visualizations


def run_experiment() -> list[dict]:
    results = []

    total_configs = len(DIMS) * len(ALGORITHMS) * len(BENCHMARKS)
    current_config = 0

    for dim in DIMS:
        print(f"Dimension: {dim}")
        for algorithm_name, algorithm_fn in ALGORITHMS.items():
            for benchmark_name, benchmark_config in BENCHMARKS.items():
                current_config += 1

                objective_fn = benchmark_config[algorithm_name]
                bounds = benchmark_config["bounds"]
                hyperparameters = HYPERPARAMETERS.copy()
                hyperparameters["num_dims"] = dim

                print(
                    f"[{current_config}/{total_configs}] Running {algorithm_name} "
                    f"on {benchmark_name}",
                )

                execution_times = []
                fitness_history = []
                for i in range(NUM_RUNS):
                    hyperparameters["seed"] = i

                    if algorithm_name == "JAX PSO":
                        hyperparameters["seed"] = random.PRNGKey(i)

                    start = time.perf_counter()
                    result = algorithm_fn(objective_fn, bounds, **hyperparameters)

                    if algorithm_name == "JAX PSO":
                        block_until_ready(result)

                    end = time.perf_counter()
                    execution_times.append(end - start)

                    _, fitness, _ = result

                    print(f"Iteration {i + 1} | Fitness: {fitness}")
                    fitness_history.append(fitness)

                mean_time = float(jnp.mean(jnp.array(execution_times)))
                std_time = float(jnp.std(jnp.array(execution_times)))

                mean_fitness = float(jnp.mean(jnp.array(fitness_history)))
                std_fitness = float(jnp.std(jnp.array(fitness_history)))

                results.extend(
                    [
                        {
                            "Dimension": dim,
                            "Benchmark": benchmark_name,
                            "Algorithm": algorithm_name,
                            "Execution Time History": execution_times,
                            "Mean of Execution Times (s)": mean_time,
                            "Standard Deviation of Execution Times (s)": std_time,
                            "Fitness History": fitness_history,
                            "Mean of Fitness": mean_fitness,
                            "Standard Deviation of Fitness": std_fitness,
                        },
                    ],
                )

    return results


if __name__ == "__main__":
    results = run_experiment()

    df = pd.DataFrame(results)
    df.to_csv("./results/experiment_results.csv")
    generate_visualizations(df)
