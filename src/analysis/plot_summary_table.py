import pandas as pd
from IPython.display import display


def plot_summary_table(df: pd.DataFrame) -> None:
    for dim in df['Dimension'].unique():
        df_dim = df[df['Dimension'] == dim]
        df_dim_execution_time = df_dim.pivot_table(
            index='Benchmark',
            columns='Algorithm',
            values ='Mean of Execution Times (s)',
        )

        df_dim_execution_time.to_csv(f'../results/tables/execution_time_table_{dim}d.csv')
        display(df_dim_execution_time)
