"""
EDA для Weibo датасета (используется в Task3)
Требуется минимум 4 графика
"""
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd
import numpy as np
from pathlib import Path

from antifraud.utils.logger import get_logger


class WeiboEDA:
    _logger = None

    @staticmethod
    def run(data, out_dir: Path):
        if WeiboEDA._logger is None:
            WeiboEDA._logger = get_logger('WeiboEDA')
        
        out_dir.mkdir(parents=True, exist_ok=True)
        
        edge_list = data.edge_index.numpy().T
        degrees = Counter(edge_list[:, 0]) + Counter(edge_list[:, 1])
        X = data.x.numpy()
        y = data.y.numpy()

        sns.set_theme(style='whitegrid')

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        ax = axes[0, 0]
        degree_values = np.array(list(degrees.values()))
        sns.histplot(degree_values[degree_values > 0], bins=50, color='steelblue', ax=ax)
        ax.set_xlabel('Degree')
        ax.set_ylabel('Frequency')
        ax.set_title('Node Degree Distribution')
        ax.set_yscale('log')

        ax = axes[0, 1]
        label_counts = pd.Series(y).value_counts().sort_index()
        label_names = ['Normal', 'Spammer']
        sns.barplot(x=label_names, y=label_counts.values, palette=['green', 'red'], ax=ax)
        for i, v in enumerate(label_counts.values):
            ax.text(i, v + label_counts.max()*0.01, f"{v} ({v/label_counts.sum()*100:.2f}%)", ha='center')
        ax.set_title('Label Distribution')
        ax.set_ylabel('Count')

        ax = axes[0, 2]
        feature_means = X.mean(axis=0)
        sns.histplot(feature_means, bins=30, color='teal', ax=ax)
        ax.set_xlabel('Mean Feature Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Feature Mean Distribution')

        ax = axes[1, 0]
        feature_vars = X.var(axis=0)
        sns.histplot(feature_vars, bins=30, color='purple', ax=ax)
        ax.set_xlabel('Feature Variance')
        ax.set_ylabel('Frequency')
        ax.set_title('Feature Variance Distribution')
        ax.set_xscale('log')

        ax = axes[1, 1]
        normal_mean = X[y == 0].mean(axis=0)
        anomaly_mean = X[y == 1].mean(axis=0)
        max_index = min(20, X.shape[1])
        ax.plot(np.arange(max_index), normal_mean[:max_index], label='Normal', marker='o', alpha=0.7)
        ax.plot(np.arange(max_index), anomaly_mean[:max_index], label='Spammer', marker='o', alpha=0.7)
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Mean Value')
        ax.set_title('Feature Mean Comparison (First 20)')
        ax.legend()

        ax = axes[1, 2]
        normal_degrees = np.array([degrees.get(i, 0) for i in range(len(y)) if y[i] == 0])
        anomaly_degrees = np.array([degrees.get(i, 0) for i in range(len(y)) if y[i] == 1])
        sns.histplot(normal_degrees, bins=30, color='green', label='Normal', ax=ax, kde=False, stat='density')
        sns.histplot(anomaly_degrees, bins=30, color='red', label='Spammer', ax=ax, kde=False, stat='density', alpha=0.7)
        ax.set_xlabel('Node Degree')
        ax.set_ylabel('Density')
        ax.set_title('Degree Distribution by Label')
        ax.set_yscale('log')
        ax.legend()
        
        fig.tight_layout()
        fig.savefig(out_dir / 'eda_weibo.png', dpi=120)
        plt.close(fig)

        WeiboEDA._logger.info("Weibo EDA saved to eda_weibo.png")
