from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def setup_styles(config: dict) -> None:
    regular = config['font_path'] / 'LibreFranklin-Regular.ttf'
    bold = config['font_path'] / 'LibreFranklin-Bold.ttf'

    fm.fontManager.addfont(str(regular))
    fm.fontManager.addfont(str(bold))

    plt.rcParams.update(config['font'])

def _save_figure(fig: plt.Figure, filename: str, config: dict) -> None:
    save_path = config['output_path'] / filename
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def plot_execution_time_bars(df: pd.DataFrame, config: dict) -> None:
    dimensions = df['Dimension'].unique()
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(7, 7),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    axes_flat = axes.flatten()

    for ax, dim in zip(axes_flat, dimensions, strict=True):
        df_dim = df[df['Dimension'] == dim]

        sns.barplot(
            data=df_dim,
            x='Mean of Execution Times (s)',
            y='Benchmark',
            hue='Algorithm',
            palette=config['palette'],
            ax=ax,
            orient='h',
        )

        ax.set_title(f'Dimension = {dim}', fontweight='bold')
        ax.set(xlabel='', ylabel='', xscale='log')
        ax.get_legend().remove()

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

    _save_figure(fig, 'execution_time_bars.pdf', config)

def plot_execution_time_dims(df: pd.DataFrame, config: dict) -> None:
    benchmarks = df['Benchmark'].unique()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7, 7),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    axes_flat = axes.flatten()

    for ax, benchmark in zip(axes_flat, benchmarks, strict=True):
        df_benchmark = df[df['Benchmark'] == benchmark]
        algorithms = df_benchmark['Algorithm'].unique()
        colors = sns.color_palette(n_colors=len(algorithms))

        sns.lineplot(
            data=df_benchmark,
            x='Dimension',
            y='Mean of Execution Times (s)',
            hue='Algorithm',
            marker='o',
            palette=config['palette'],
            ax=ax,
        )

        for algorithm, color in zip(algorithms, colors, strict=True):
            data_algorithm = df_benchmark[df_benchmark['Algorithm'] == algorithm]
            mean = data_algorithm['Mean of Execution Times (s)']
            std = data_algorithm['Standard Deviation of Execution Times (s)']

            ax.fill_between(
                data_algorithm['Dimension'],
                mean - std,
                mean + std,
                color=color,
                alpha=0.2,
            )

        ax.set_title(f'{benchmark}', fontweight='bold')
        ax.set(xlabel='', ylabel='', yscale='log')
        ax.get_legend().remove()

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

    _save_figure(fig, 'execution_time_dims.pdf', config)

def plot_speedup_heatmap(df: pd.DataFrame, config: dict) -> None:
    dimensions = df['Dimension'].unique()
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(7, 7),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    axes_flat = axes.flatten()

    for i, (ax, dim) in enumerate(zip(axes_flat, dimensions, strict=True)):
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
            cmap=config['palette'],
            ax=ax,
            cbar=(i in {2, 5, 8}),
            cbar_kws={'label': 'Speedup Factor'} if i in {5} else {},
        )

        ax.set_title(f'Dimension = {dim}', fontweight='bold')
        ax.set(xlabel='', ylabel='')

    fig.supxlabel('Reference Algorithm')
    fig.supylabel('Target Algorithm')

    _save_figure(fig, 'speedup_heatmap.pdf', config)

def generate_summary_tables(df: pd.DataFrame, config: dict) -> None:
    for dim, df_dim in df.groupby('Dimension'):
        pivot_table = df_dim.pivot_table(
            index='Benchmark',
            columns='Algorithm',
            values ='Mean of Execution Times (s)',
        )
        save_path = config['output_path'] / f'execution_time_table_{dim}d.csv'
        pivot_table.to_csv(save_path)


def generate_visualizations(df: pd.DataFrame) -> None:
    config = {
        'font_path': Path('./fonts/Libre_Franklin/static/'),
        'output_path': Path('./results/'),
        'palette': 'viridis',
        'font': {
            'font.size': 7,
            'axes.titlesize': 9,
            'legend.fontsize': 8,
            'axes.labelsize': 10,
            'xtick.labelsize': 7.5,
            'ytick.labelsize': 7.5,
            'font.family': 'Libre Franklin',
        },
    }

    setup_styles(config)

    print('Plotting execution time bars...')
    plot_execution_time_bars(df, config)

    print('Plotting execution time by dimensions...')
    plot_execution_time_dims(df, config)

    print('Plotting speedup heatmap...')
    plot_speedup_heatmap(df, config)

    print('Generating summary tables...')
    generate_summary_tables(df, config)

if __name__ == '__main__':
    df = pd.read_csv('./results/experiment_results.csv')
    generate_visualizations(df)
