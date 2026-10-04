import pandas as pd
from pathlib import Path
from itertools import product
import matplotlib.pyplot as plt

base_dir = Path(__file__).resolve().parent

project_root = base_dir.parent
metrics_proc = project_root / 'results' / 'metrics' / 'metrics.csv'
analysis_proc = project_root / 'results' / 'analysis'

analysis_proc.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(metrics_proc)

model = df['model'].unique()
language = df['language'].unique()
prompt_type = df['prompt_type'].unique()

for metric in ['mean_depth', 'max_depth', 'mdd', 'length']:
    table=pd.pivot_table(
    data=df,
    values=metric, 
    index='temperature',
    columns=['model', 'language', 'prompt_type'],
    aggfunc='median'
)
    colors = {('english', 'formal'): '#58a6ff', ('english', 'casual'): '#3fb950',  ('french', 'formal'): '#808080', ('french', 'casual'): '#FF0000'}
    styles = {'formal': '-', 'casual': '--'}
    
    for model_name in model:
        for lang, p_type in product(language, prompt_type):
            plt.plot(
            table.index,
            table[model_name][lang][p_type],
            color=colors[(lang, p_type)],
            linestyle=styles[p_type],
            label=f"{lang} {p_type}",
            )
        plt.title(model_name)
        plt.xlabel('temperature')
        plt.ylabel(metric)
        plt.legend()
        plt.xticks([0.0, 0.4, 0.8, 1.2, 1.6, 2.0])
        safe_name = model_name.replace('/', '_').replace(':', '_')
        plt.savefig(analysis_proc / f'{metric}_{safe_name}.pdf')
        plt.close()
           