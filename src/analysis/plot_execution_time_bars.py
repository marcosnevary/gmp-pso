import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_execution_time_bars(df: pd.DataFrame) -> None:
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'legend.fontsize': 7})

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(7.2, 9.5),
        constrained_layout=True,
        sharex=True,
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
            f'Execution Time Comparison ({dim}D)',
            loc='right',
            fontweight='bold',
        )

        ax.set_xscale('log')
        ax.set_xlabel('Execution Time (s)')
        ax.set_ylabel('')

        ax.get_legend().remove()
        sns.despine(left=True, bottom=True)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )

    fig.savefig(
        '../results/plots/execution-time-bars/execution_time_bars.pdf',
        dpi=300,
        bbox_inches='tight',
    )

    plt.show()
