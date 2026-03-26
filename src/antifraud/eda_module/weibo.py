"""
EDA для Weibo датасета (используется в Task3)
Требуется минимум 4 графика
"""
import matplotlib.pyplot as plt
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

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Распределение степеней узлов
        ax = axes[0, 0]
        degree_values = list(degrees.values())
        ax.hist(degree_values, bins=50, color='steelblue', alpha=0.7)
        ax.set_xlabel('Degree')
        ax.set_ylabel('Frequency')
        ax.set_title('Node Degree Distribution')
        ax.set_yscale('log')
        
        # 2. Распределение меток (0 - normal, 1 - anomaly)
        ax = axes[0, 1]
        label_counts = pd.Series(y).value_counts()
        ax.pie(label_counts, labels=['Normal', 'Spammer'], autopct='%1.1f%%',
               colors=['green', 'red'], explode=[0, 0.1])
        ax.set_title('Label Distribution (Normal vs Spammer)')
        
        # 3. Распределение признаков (среднее значение по узлам)
        ax = axes[0, 2]
        feature_means = X.mean(axis=0)
        ax.hist(feature_means, bins=30, color='teal', alpha=0.7)
        ax.set_xlabel('Mean Feature Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Feature Mean Distribution')
        
        # 4. Распределение дисперсии признаков
        ax = axes[1, 0]
        feature_vars = X.var(axis=0)
        ax.hist(feature_vars, bins=30, color='purple', alpha=0.7)
        ax.set_xlabel('Feature Variance')
        ax.set_ylabel('Frequency')
        ax.set_title('Feature Variance Distribution')
        
        # 5. Сравнение признаков для normal vs anomaly узлов
        ax = axes[1, 1]
        normal_mean = X[y == 0].mean(axis=0)
        anomaly_mean = X[y == 1].mean(axis=0)
        ax.plot(normal_mean[:20], label='Normal', alpha=0.7)
        ax.plot(anomaly_mean[:20], label='Spammer', alpha=0.7)
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Mean Value')
        ax.set_title('Feature Comparison (First 20)')
        ax.legend()
        
        # 6. Количество связей для normal vs anomaly
        ax = axes[1, 2]
        normal_degrees = [degrees.get(i, 0) for i in range(len(y)) if y[i] == 0]
        anomaly_degrees = [degrees.get(i, 0) for i in range(len(y)) if y[i] == 1]
        ax.hist([normal_degrees, anomaly_degrees], bins=30, alpha=0.6, 
                label=['Normal', 'Spammer'], color=['green', 'red'])
        ax.set_xlabel('Node Degree')
        ax.set_ylabel('Frequency')
        ax.set_title('Degree Distribution by Label')
        ax.legend()
        
        fig.tight_layout()
        fig.savefig(out_dir / 'eda_weibo.png', dpi=100)
        plt.close(fig)

        WeiboEDA._logger.info("Weibo EDA saved to eda_weibo.png")
