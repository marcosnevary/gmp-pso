import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_execution_time_boxplot(df: pd.DataFrame) -> None:
    _, axes = plt.subplots(3, 3, figsize=(15, 3 * 4))
    axes_flat = axes.flatten()

    for i, dim in enumerate(df['Dimension'].unique()):
        ax = axes_flat[i]
        df_dim = df[df['Dimension'] == dim]

        sns.boxplot(
            data=df_dim,
            x='Benchmark',
            y='Execution Times',
            hue='Algorithm',
            palette='viridis',
            ax=ax,
        )

        ax.legend(
            loc='upper left',
            bbox_to_anchor=(-0.02, 1.12),
            ncol=3,
            frameon=False,
            fontsize=10,
        )

        plt.title(
            f'Execution Time Distribution ({dim}d)',
            pad=30,
            loc='left',
            fontsize=14,
            fontweight='bold',
        )

        ax.set_yscale('log')
        sns.despine(left=True, bottom=True)

    plt.tight_layout()
    plt.savefig(
        f'../results/plots/execution-time-boxplot/execution_time_boxplot_{dim}d.png',
        dpi=300,
        bbox_inches='tight',
    )
    plt.show()

