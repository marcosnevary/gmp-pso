import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_execution_time_bars(df: pd.DataFrame) -> None:
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

        sns.barplot(
            data=df_dim,
            x='Mean of Execution Times (s)',
            y='Benchmark',
            hue='Algorithm',
            palette='viridis',
            errorbar=None,
            ax=ax,
            orient='h',
        )

        ax.set_title(
            f'Dimension = {dim}',
            loc='left',
            fontweight='bold',
        )

        ax.set_xscale('log')
        ax.set_xlabel('')
        ax.set_ylabel('')

        ax.get_legend().remove()
        sns.despine(ax=ax)

    fig.supxlabel('Mean of Execution Times (s)', fontsize=8)
    fig.supylabel('Benchmarks', fontsize=8)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
        frameon=False,
    )

    fig.savefig(
        '../results/plots/execution-time-bars/execution_time_bars.pdf',
        dpi=300,
        bbox_inches='tight',
    )

    plt.show()
