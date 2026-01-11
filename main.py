import time

import jax.numpy as jnp
import pandas as pd
from jax import block_until_ready

from gmp_pso.benchmarks import ALGORITHMS, BENCHMARKS, DIMS, HYPERPARAMETERS, NUM_RUNS
from gmp_pso.plot_benchmarks import generate_visualizations


def run_experiment() -> list[dict]:
    results = []

    total_configs = len(DIMS) * len(ALGORITHMS) * len(BENCHMARKS)
    current_config = 0

    for dim in DIMS:
        print(f'Dimension: {dim}')
        for algorithm_name, algorithm_fn in ALGORITHMS.items():
            for benchmark_name, benchmark_config in BENCHMARKS.items():
                current_config += 1

                objective_fn = benchmark_config[algorithm_name]
                bounds = benchmark_config['bounds']
                hyperparameters = HYPERPARAMETERS.copy()
                hyperparameters['num_dims'] = dim

                print(
                    f'[{current_config}/{total_configs}] Running {algorithm_name} '
                    f'on {benchmark_name}',
                )

                execution_times = []
                for _ in range(NUM_RUNS):
                    start = time.perf_counter()
                    result = algorithm_fn(objective_fn, bounds, **hyperparameters)

                    if algorithm_name == 'JAX PSO':
                        block_until_ready(result)

                    end = time.perf_counter()
                    execution_times.append(end - start)

                mean_time = float(jnp.mean(jnp.array(execution_times[1:])))
                std_time = float(jnp.std(jnp.array(execution_times[1:])))

                results.extend(
                    [
                        {
                            'Dimension': dim,
                            'Benchmark': benchmark_name,
                            'Algorithm': algorithm_name,
                            'Execution Times': execution_times,
                            'Mean of Execution Times (s)': mean_time,
                            'Standard Deviation of Execution Times (s)': std_time,
                        },
                    ],
                )

    return results

if __name__ == '__main__':
    results = run_experiment()

    df = pd.DataFrame(results)
    df.to_csv('./results/experiment_results.csv')
    generate_visualizations(df)
