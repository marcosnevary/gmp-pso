import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_speedup_heatmap(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(7, 7),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
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
            annot_kws={'size': 6},
            fmt='.1f',
            cmap='viridis',
            ax=ax,
            cbar=(i in {2, 5, 8}),
            cbar_kws={'label': 'Speedup Factor'},
        )

        ax.set_title(f'Dimension = {dim}', loc='left', fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')

    fig.supxlabel('Reference Algorithm')
    fig.supylabel('Target Algorithm')

    fig.savefig(
        '../results/plots/speedup-heatmap/speedup_heatmap.pdf',
        dpi=300,
        bbox_inches='tight',
    )
    plt.show()
