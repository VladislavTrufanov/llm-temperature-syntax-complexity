import re
import pandas
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from pathlib import Path

base_dir = Path(__file__).resolve().parent

project_root = base_dir.parent
metrics_proc = project_root / 'results' / 'metrics' / 'metrics.csv'
analysis_proc = project_root / 'results' / 'analysis'

analysis_proc.mkdir(parents=True, exist_ok=True)

df = pandas.read_csv(metrics_proc)

df['model'] = df['file_name'].str.rsplit('_', n=2).str[0]
df['language'] = df['file_name'].str.rsplit('_', n=2).str[1]
df['prompt_type'] = df['file_name'].str.rsplit('_', n=2).str[2]

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
    
    for model_name in df['model'].unique():
        safe=re.sub(r'[^\w\-]', '_', model_name)
        for (language, prompt_type) in table[model_name].columns:
            plt.plot(
            table.index,
            table[model_name][(language, prompt_type)],
            color=colors[(language, prompt_type)],
            linestyle=styles[prompt_type],
            label=f"{language} {prompt_type}",
        )
        plt.title(model_name)
        plt.xlabel('temperature')
        plt.ylabel(metric)
        plt.legend()
        plt.xticks([0.0, 0.4, 0.8, 1.2, 1.6, 2.0])
        plt.savefig(analysis_proc / f'{metric}_{safe}.pdf')
        plt.close()
           