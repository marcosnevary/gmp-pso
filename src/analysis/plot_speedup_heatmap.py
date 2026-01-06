import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_speedup_heatmap(df: pd.DataFrame) -> None:
    _, axes = plt.subplots(3, 3, figsize=(20, 5 * 4))
    axes_flat = axes.flatten()

    for i, dim in enumerate(df['Dimension'].unique()):
        ax = axes_flat[i]
        df_dim = df[df['Dimension'] == dim]

        algorithm_means = []

        for algorithm in df_dim['Algorithm'].unique():
            time_algorithm = df_dim[
                (df_dim['Algorithm'] == algorithm)
            ]['Mean of Execution Times (s)'].mean()
            algorithm_means.append((algorithm, time_algorithm))

        speedups = []

        for algorithm, mean_time in algorithm_means:
            for algorithm2, mean_time2 in algorithm_means:
                speedup = mean_time2 / mean_time
                speedups.append(
                    {
                        'Algorithm 1': algorithm,
                        'Algorithm 2': algorithm2,
                        'Speedup': speedup,
                    },
                )

        df_speedup = pd.DataFrame(speedups).pivot_table(
            index='Algorithm 1',
            columns='Algorithm 2',
            values='Speedup',
        )

        sns.heatmap(
            data=df_speedup,
            annot=True,
            fmt='.2f',
            cmap='viridis',
            ax=ax,
        )

        plt.title(
            f'Speedup Heatmap ({dim}d)',
            pad=30,
            loc='left',
            fontsize=14,
            fontweight='bold',
        )

    plt.tight_layout()
    plt.savefig(
        f'../results/plots/speedup-heatmap/speedup_heatmap_{dim}d.png',
        dpi=300,
        bbox_inches='tight',
    )
    plt.show()
