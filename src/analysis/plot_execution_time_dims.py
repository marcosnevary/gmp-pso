import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_execution_time_dims(df: pd.DataFrame) -> None:
    _, axes = plt.subplots(3, 3, figsize=(20, 5 * 4))
    axes_flat = axes.flatten()

    for i, benchmark in enumerate(df['Benchmark'].unique()):
        ax = axes_flat[i]
        df_benchmark = df[df['Benchmark'] == benchmark]

        sns.lineplot(
            data=df_benchmark,
            x='Dimension',
            y='Mean of Execution Times (s)',
            hue='Algorithm',
            marker='o',
            palette='viridis',
            ax=ax,
        )

        ax.set_title(
            f'Execution Time Comparison ({benchmark})',
            pad=30,
            loc='left',
            fontsize=14,
            fontweight='bold',
        )

        ax.legend(
            loc='upper left',
            bbox_to_anchor=(-0.02, 1.07),
            ncol=3,
            frameon=False,
            fontsize=10,
        )

        ax.set_yscale('log')
        sns.despine(left=True)

    plt.tight_layout()
    plt.savefig(
        f'../results/plots/execution-time-dims/execution_time_dims_{benchmark}.png',
        dpi=300,
        bbox_inches='tight',
    )
    plt.show()
