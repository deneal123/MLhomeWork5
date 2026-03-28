"""
EDA для BankSim датасета (используется в Task1 и Task4)
Требуется минимум 4 графика
"""
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import pandas as pd

from antifraud.utils.logger import get_logger


class BankSimEDA:
    _logger = None

    @staticmethod
    def run(df, out_dir: Path):
        if BankSimEDA._logger is None:
            BankSimEDA._logger = get_logger('BankSimEDA')
        
        out_dir.mkdir(parents=True, exist_ok=True)

        sns.set_theme(style='whitegrid')
        palette = {'Normal': 'green', 'Fraud': 'red'}

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        ax = axes[0, 0]
        fraud_counts = df['fraud'].value_counts().sort_index()
        classes = ['Normal', 'Fraud']
        fps = fraud_counts.values
        ax.bar(classes, fps, color=[palette[c] for c in classes])
        tot = fps.sum()
        for i, v in enumerate(fps):
            ax.text(i, v + tot*0.01, f"{v} ({v/tot*100:.2f}%)", ha='center')
        ax.set_title('Class Distribution (Fraud vs Normal)')
        ax.set_ylabel('Count')
        
        ax = axes[0, 1]
        for cls, color in [('Normal', 'green'), ('Fraud', 'red')]:
            cls_val = 0 if cls == 'Normal' else 1
            sns.histplot(df[df['fraud'] == cls_val]['amount'].clip(upper=df['amount'].quantile(0.99)), bins=60, kde=True, stat='density', element='step', fill=False, label=cls, color=color, ax=ax)
        ax.set_xlabel('Amount (trimmed 99% quantile)')
        ax.set_ylabel('Density')
        ax.set_title('Amount Density by Class')
        ax.legend()

        ax = axes[0, 2]
        cat_fraud = df[df['fraud'] == 1]['category'].value_counts().head(10).sort_values()
        cat_fraud.plot(kind='barh', ax=ax, color='red')
        ax.set_title('Top 10 Fraud Categories')
        ax.set_xlabel('Count')

        ax = axes[1, 0]
        age_series = pd.to_numeric(df['age'], errors='coerce').dropna()
        if age_series.empty:
            ax.text(0.5, 0.5, 'No age data available', ha='center', va='center')
            ax.set_title('Age Distribution')
            ax.set_xlabel('Age')
            ax.set_ylabel('Count')
        else:
            age_min = int(age_series.min())
            age_max = int(age_series.max())
            age_bins = np.arange(age_min, age_max + 5, 5)
            sns.histplot(age_series, bins=age_bins, color='steelblue', ax=ax)
            ax.set_title('Age Distribution')
            ax.set_xlabel('Age')
            ax.set_ylabel('Count')

        ax = axes[1, 1]
        gender_fraud = df.groupby(['gender', 'fraud']).size().unstack(fill_value=0)
        gender_fraud_norm = gender_fraud.div(gender_fraud.sum(axis=1), axis=0)*100
        gender_fraud_norm.plot(kind='bar', ax=ax, color=['green', 'red'])
        ax.set_title('Gender vs Fraud Rate (%)')
        ax.set_xlabel('Gender')
        ax.set_ylabel('Percent')
        ax.legend(['Normal', 'Fraud'])

        ax = axes[1, 2]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_with_fraud = df[numeric_cols].corr()['fraud'].drop('fraud').abs().sort_values(ascending=True)
        corr_with_fraud.plot(kind='barh', ax=ax, color='teal')
        ax.set_title('Absolute Correlation with Fraud')
        ax.set_xlabel('Pearson |r|')
        
        fig.tight_layout()
        fig.savefig(out_dir / 'eda_banksim.png', dpi=120)
        plt.close(fig)

        BankSimEDA._logger.info("BankSim EDA saved to eda_banksim.png")
