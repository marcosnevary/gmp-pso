import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_execution_time_dims(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(7, 7),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
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
            ax=ax
        )

        algorithms = df_benchmark['Algorithm'].unique()
        colors = sns.color_palette('viridis', n_colors=len(algorithms))

        for i, algorithm in enumerate(algorithms):
            data_algorithm = df_benchmark[df_benchmark['Algorithm'] == algorithm]
            
            ax.fill_between(
                data_algorithm['Dimension'],
                data_algorithm['Mean of Execution Times (s)'] - data_algorithm['Standard Deviation of Execution Times (s)'],
                data_algorithm['Mean of Execution Times (s)'] + data_algorithm['Standard Deviation of Execution Times (s)'],
                color=colors[i],
                alpha=0.2
            )

        ax.set_title(
            f'{benchmark}',
            loc='left',
            fontweight='bold',
        )

        ax.set_yscale('log')
        ax.set_xlabel('')
        ax.set_ylabel('')

        ax.get_legend().remove()
        sns.despine(left=True)

    fig.supxlabel('Dimension', fontsize=8)
    fig.supylabel('Mean of Execution Times (s)', fontsize=8)

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
        '../results/plots/execution-time-dims/execution_time_dims.pdf',
        dpi=300,
        bbox_inches='tight',
    )
    plt.show()
