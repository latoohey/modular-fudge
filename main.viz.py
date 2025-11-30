import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import pi
import pandas as pd
from pathlib import Path
import sys
import os
import random


class StyleVisualizer:
    def __init__(self, df_results, output_dir=None):
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        pd.set_option('display.max_rows', None)
        self.df = df_results

        # Set publication-ready aesthetics
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'

    def save_desc_stats(self, drop_errors, filename):
        if drop_errors:
            desc_stats = self.df[self.df['error'] < 0.5].groupby('model_name').describe().T
        else:
            desc_stats = self.df.groupby('model_name').describe().T
        desc_stats.to_csv(f'{self.output_dir}/{filename}')

    def save_plot(self, filename):
        path = self.output_dir / filename
        plt.savefig(path)
        plt.show()
        print(f"Saved plot to {path}")
        plt.close()


    def plot_relative_changes(self,
                          baseline_name,
                          metrics,
                          model_col='model_name',
                          ax_labels=None, # <--- Pass your pretty list here e.g. ["Reading Ease", "Perplexity"]
                          ylabel="% Change",
                          xlabel="Metrics",
                          title="Relative Performance vs Llama 3B Instruct",
                          drop_errors=True,
                          model_label_map=None,
                          custom_colors=None):

          line_color = "#2C3E50"

          # 1. Data Prep
          df = self.df.copy()
          if drop_errors:
              df = df[df['error'] < 0.5]

          means = df.groupby(model_col)[metrics].mean()

          if baseline_name not in means.index:
              print(f"Error: Baseline '{baseline_name}' not found.")
              return

          baseline_vals = means.loc[baseline_name]
          delta = ((means - baseline_vals) / baseline_vals) * 100
          delta = delta.drop(baseline_name)

          # 2. Reshape (Melt)
          delta_reset = delta.reset_index()
          long_df = delta_reset.melt(id_vars=model_col,
                                    value_vars=metrics,
                                    var_name='Metric_Name',
                                    value_name='Percent_Change')

          # --- Renaming Models (Using your map) ---
          if model_label_map:
              long_df[model_col] = long_df[model_col].replace(model_label_map)

          # --- NEW: Renaming Metrics (Using ax_labels) ---
          # We map the ugly column names to your pretty list
          if ax_labels and len(ax_labels) == len(metrics):
              # Create a dictionary: {'ugly_name': 'Pretty Label', ...}
              metric_map = dict(zip(metrics, ax_labels))
              # Update the column directly
              long_df['Metric_Name'] = long_df['Metric_Name'].map(metric_map)

          # 3. Plotting
          plt.figure(figsize=(10, 6))

          ax = sns.barplot(
              data=long_df,
              x='Metric_Name', # Now uses the pretty names
              y='Percent_Change',
              hue=model_col,
              palette=custom_colors,
              edgecolor='white',
              linewidth=1
          )

          plt.axhline(0, color='black', linewidth=1.5)

          # 4. Styling
          ax.set_facecolor('white')
          ax.yaxis.grid(True, color='#ECECEC', linestyle='-', linewidth=1)
          ax.xaxis.grid(False)
          ax.set_axisbelow(True)
          sns.despine(left=True, bottom=True, right=True, top=True)

          plt.xlabel(xlabel, size=12, color=line_color, fontweight='bold', labelpad=10)
          plt.ylabel(ylabel, size=12, color=line_color, fontweight='bold')
          plt.title(title, fontsize=14, fontweight='bold', color=line_color, pad=20)

          # Tick styling
          plt.xticks(fontsize=11, color=line_color, fontweight='bold')
          plt.yticks(color=line_color)
          ax.tick_params(axis='y', which='both', length=0)
          ax.tick_params(axis='x', which='both', length=0, pad=5)

          # 5. Add Value Labels
          for container in ax.containers:
              ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=9)

          plt.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
          plt.tight_layout()

          self.save_plot(f"grouped_delta_vs_{baseline_name}.png")

    def plot_box_plots(self,
                      metric,
                      models,
                      ylabel,
                      xlabel,
                      title,
                      filename,
                      drop_errors=True,
                      show_outliers=False,
                      model_label_map=None,
                      custom_colors=None):

        box_color = "#FF4B3E" # Keep for fallbacks
        line_color = "#2C3E50"

        # 1. Data Prep
        df = self.df.copy()
        if drop_errors:
            df = df[df['error'] < 0.5]

        # 2. Apply Renaming (Single Source of Truth)
        # This aligns the dataframe with your master config
        plotting_order = models.copy() # Start with raw IDs

        if model_label_map:
            # Rename the actual data
            df['model_name'] = df['model_name'].replace(model_label_map)

            # We must also rename the 'order' list so Seaborn can find the data
            # syntax: look up the new name, or keep the old one if not found
            plotting_order = [model_label_map.get(m, m) for m in models]

        # 3. Setup Figure
        plt.figure(figsize=(6, 8)) # Slightly wider to accommodate labels if needed

        ax = sns.boxplot(
            x='model_name',
            y=metric,
            data=df,
            order=plotting_order, # Use the updated order list

            # 4. Apply Custom Colors
            palette=custom_colors,

            # Styling
            width=0.4,
            linewidth=1.5,
            showfliers=show_outliers,
            showcaps=True,

            # Updated props to work with custom palettes
            # Note: We remove 'edgecolor=box_color' from boxprops so it uses the palette color
            # or we set it to 'face' to match the fill, or a neutral color.
            boxprops=dict(linewidth=0, alpha=0.9),
            medianprops=dict(color="white", linewidth=2.5),
            whiskerprops=dict(color=line_color),
            capprops=dict(color=line_color, linewidth=1.5),
            flierprops=dict(marker='o', markerfacecolor=line_color, markeredgecolor=line_color, markersize=6)
        )

        # Standard styling adjustments
        ax.set_facecolor('white')
        ax.yaxis.grid(True, color='#ECECEC', linestyle='-', linewidth=1)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        sns.despine(left=True, bottom=True, right=True, top=True)

        plt.xlabel(xlabel, size=12, color=line_color, fontweight='bold', labelpad=10)
        plt.ylabel(ylabel, size=12, color=line_color, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold', color=line_color, pad=20)

        # Let Seaborn handle the labels, just style them
        plt.xticks(fontsize=11, color=line_color, fontweight='bold', ha='center')
        plt.yticks(color=line_color)

        ax.tick_params(axis='y', which='both', length=0)
        ax.tick_params(axis='x', which='both', length=0, pad=5)

        self.save_plot(filename)
        
def mount_colab_helper(project_path, judge_path=None):
    """Helper to handle the specific pathing needs of Colab."""
    if 'google.colab' in sys.modules:
        from google.colab import drive
        drive.mount('/content/drive')
        base = '/content/drive/My Drive/'

        # Return updated paths
        p_path = f"{base}{project_path}"
        j_path = f"{base}{judge_path}" if judge_path else None
        return p_path, j_path
    return project_path, judge_path

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Global seed set to {seed}")
    
    
if __name__ == '__main__':
    
    PROJECT_PATH_IN = "modular-fudge/data/test_results"
    SEED = 24601
    seed_everything(SEED)
    full_project_path, _ = mount_colab_helper(PROJECT_PATH_IN)
    
    df_results = pd.read_csv(f"{full_project_path}/combined_evals.csv")

    df_results.loc[df_results['flesch_kincaid_grade'] > 30, 'error'] = 1

    #Initialize Visualizer
    viz = StyleVisualizer(df_results, output_dir=f"{full_project_path}/plots")

    models_to_compare = ['baseline', 'lstm_2_256', 'mamba_128_4_16_1']

    available_models = df_results['model_name'].unique()
    valid_models = [m for m in models_to_compare if m in available_models]



    viz.save_desc_stats(True, 'desc_stats_wo_errors.csv')
    viz.save_desc_stats(False, 'desc_stats_w_errors.csv')

    # --- 1. Define The "Truth" Once ---

    # Map RAW IDs -> PRETTY LABELS
    # (Ensure every model ID in your data is represented here)
    MY_LABEL_MAP = {
        'lstm_2_256': 'LSTM',
        'mamba_128_4_16_1': 'Mamba',
        'baseline': 'Llama 3B Instruct'
    }

    # Map PRETTY LABELS -> COLORS
    # (Use the values from the dict above as keys here)
    MY_COLOR_MAP = {
        'LSTM': '#3498db',    # Blue
        'Mamba': '#e67e22',  # Orange
        'Llama 3B Instruct': '#95a5a6'        # Grey
    }

    # Define the Raw Order you want them to appear in
    # (The functions will translate these automatically)
    RAW_ORDER = ['baseline', 'lstm_2_256', 'mamba_128_4_16_1']


    # --- 2. Call Your Functions ---

    # Plot A: Box Plot
    viz.plot_box_plots(
        metric='flesch_reading_ease',
        models=RAW_ORDER,
        show_outliers=False,
        xlabel='',
        ylabel='Reading Ease',
        drop_errors=True,
        title='Flesch Reading Ease',
        filename='box_reading_ease.png',
        model_label_map=MY_LABEL_MAP, # <--- Pass Config
        custom_colors=MY_COLOR_MAP    # <--- Pass Config
    )

    viz.plot_box_plots(
        metric='flesch_kincaid_grade',
        models=RAW_ORDER,
        show_outliers=False,
        xlabel='',
        ylabel='Grade Level',
        drop_errors=True,
        title='Flesch-Kincaid Grade Level',
        filename='box_grade_level.png',
        model_label_map=MY_LABEL_MAP, # <--- Pass Config
        custom_colors=MY_COLOR_MAP    # <--- Pass Config
    )

    # Plot B: Relative Changes (Grouped Bar)
    viz.plot_relative_changes(
        baseline_name='baseline',
        drop_errors=True,
        metrics=['avg_sent_length', 'complex_sentence_ratio'],
        ax_labels=['Sentence Length', 'Complex Sentence Ratio'],
        xlabel='Sentence Metrics',
        model_label_map=MY_LABEL_MAP, # <--- Same Config
        custom_colors=MY_COLOR_MAP    # <--- Same Config
    )