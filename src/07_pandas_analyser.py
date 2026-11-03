import pandas as pd
from itertools import product
import matplotlib.pyplot as plt

from common import METRICS_DIR, ANALYSIS_DIR, safe_name

# median columns from cell_summary.csv
METRICS = [
    'mean_depth_med_median',
    'max_depth_med_median',
    'mdd_med_median',
    'length_med_median']

COLORS = {
    ('english', 'formal'): '#58a6ff',
    ('english', 'casual'): '#3fb950',
    ('french', 'formal'): '#808080',
    ('french', 'casual'): '#FF0000',
}

STYLES = {'formal': '-', 'casual': '--'}
XTICKS = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]


def build_pivot_table(df, metric):
    """Pivot to temperature rows x (model, language, prompt_type) columns."""
    return pd.pivot_table(
        data=df,
        values=metric,
        index='temperature',
        columns=['model', 'language', 'prompt_type'],
        aggfunc='mean',
    )


def compute_ylim(table, languages, prompt_types, margin=0.05):
    """Return the y-range shared by all plots of a metric, so models stay comparable."""
    values = []
    for model_name in table.columns.get_level_values('model').unique():
        for lang, p_type in product(languages, prompt_types):
            values.extend(table[model_name][lang][p_type].dropna().tolist())
    low, high = min(values), max(values)
    pad = (high - low) * margin if high > low else 0.0
    return low - pad, high + pad


def plot_model(table, metric, model_name, languages, prompt_types, output_path, ylim=None):
    """Plot one model: a line per language x prompt_type over temperature."""
    for lang, p_type in product(languages, prompt_types):
        plt.plot(
            table.index,
            table[model_name][lang][p_type],
            color=COLORS[(lang, p_type)],
            linestyle=STYLES[p_type],
            label=f'{lang} {p_type}',
        )
    plt.title(model_name)
    plt.xlabel('temperature')
    plt.ylabel(metric)
    plt.legend()
    plt.xticks(XTICKS)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.savefig(output_path)
    plt.close()


def main():
    """Plot each metric from cell_summary.csv as one PDF per model."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(METRICS_DIR / 'cell_summary.csv')
    models = df['model'].unique()
    languages = df['language'].unique()
    prompt_types = df['prompt_type'].unique()

    for metric in METRICS:
        table = build_pivot_table(df, metric)
        ylim = compute_ylim(table, languages, prompt_types)
        for model_name in models:
            plot_model(
                table,
                metric,
                model_name,
                languages,
                prompt_types,
                ANALYSIS_DIR / f'{metric}_{safe_name(model_name)}.pdf',
                ylim=ylim
            )


if __name__ == '__main__':
    main()