from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _save_figure(fig: plt.Figure, filename: str, config: dict) -> None:
    save_path = config["output_path"] / filename
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_execution_time_dims(df: pd.DataFrame, config: dict) -> None:
    benchmarks = df["Benchmark"].unique()
    fig, axes = plt.subplots(
        4,
        4,
        figsize=(7, 7),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    axes_flat = axes.flatten()

    for ax, benchmark in zip(axes_flat, benchmarks, strict=True):
        df_benchmark = df[df["Benchmark"] == benchmark]
        algorithms = df_benchmark["Algorithm"].unique()
        colors = sns.color_palette(n_colors=len(algorithms))

        sns.lineplot(
            data=df_benchmark,
            x="Dimension",
            y="Mean of Execution Times (s)",
            hue="Algorithm",
            marker="o",
            palette=config["palette"],
            ax=ax,
        )

        for algorithm, color in zip(algorithms, colors, strict=True):
            data_algorithm = df_benchmark[df_benchmark["Algorithm"] == algorithm]
            mean = data_algorithm["Mean of Execution Times (s)"]
            std = data_algorithm["Standard Deviation of Execution Times (s)"]

            ax.fill_between(
                data_algorithm["Dimension"],
                mean - std,
                mean + std,
                color=color,
                alpha=0.2,
            )

        ax.set_title(f"{benchmark}", fontweight="bold")
        ax.set(xlabel="", ylabel="", yscale="log")
        ax.get_legend().remove()

    fig.supxlabel("Dimension", fontsize=8)
    fig.supylabel("Mean of Execution Times (s)", fontsize=8)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
        frameon=False,
    )

    _save_figure(fig, "execution_time_dims.pdf", config)


def plot_convergence(df: pd.DataFrame, config: dict) -> None:
    benchmarks = df["Benchmark"].unique()
    dimensions = df["Dimension"].unique()

    for benchmark in benchmarks:
        for dimension in dimensions:
            df_subset = df[(df["Dimension"] == dimension) & (df["Benchmark"] == benchmark)]

            fig, ax = plt.subplots()

            for _, row in df_subset.iterrows():
                sns.lineplot(
                    data=row["Fitness History"],
                    ax=ax,
                    label=row["Algorithm"],
                )

            ax.set(xlabel="Iteration", ylabel="Fitness")
            ax.set_title("Convergence History", fontweight="bold")
            ax.legend(title="Algorithms")
            _save_figure(fig, f"convergence_{benchmark}_{dimension}d.pdf", config)


def generate_summary_tables(df: pd.DataFrame, config: dict) -> None:
    for dim, df_dim in df.groupby("Dimension"):
        pivot_table = df_dim.pivot_table(
            index="Benchmark",
            columns="Algorithm",
            values="Mean of Execution Times (s)",
        )
        save_path = config["output_path"] / f"execution_time_table_{dim}d.csv"
        pivot_table.to_csv(save_path)


def generate_visualizations(df: pd.DataFrame) -> None:
    config = {
        "output_path": Path("./results/"),
        "palette": "viridis",
        "font": {
            "font.size": 7,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "axes.labelsize": 10,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
        },
    }

    print("Plotting execution time by dimensions...")
    plot_execution_time_dims(df, config)

    print("Generating summary tables...")
    generate_summary_tables(df, config)

    print("Plotting convergence histories...")
    plot_convergence(df, config)
