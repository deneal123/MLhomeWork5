"""
EDA для CreditCard датасета (используется в Task2)
Требуется минимум 4 графика
"""
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from antifraud.utils.logger import get_logger


class CreditCardEDA:
    _logger = None

    @staticmethod
    def run(df, out_dir: Path):
        if CreditCardEDA._logger is None:
            CreditCardEDA._logger = get_logger('CreditCardEDA')
        
        df = df.copy()
        out_dir.mkdir(parents=True, exist_ok=True)
        
        required_cols = ['Class', 'Time', 'Amount']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            CreditCardEDA._logger.warning(f"Missing columns: {missing}. Available: {df.columns.tolist()}")
            return

        sns.set_theme(style='whitegrid')
        palette = {'Normal': 'green', 'Fraud': 'red'}
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        ax = axes[0, 0]
        class_counts = df['Class'].value_counts().sort_index()
        class_labels = ['Normal', 'Fraud']
        class_pct = (class_counts / class_counts.sum() * 100).round(2)
        sns.barplot(x=class_labels, y=class_counts.values, palette=[palette[label] for label in class_labels], ax=ax)
        for i, (count, pct) in enumerate(zip(class_counts.values, class_pct.values)):
            ax.text(i, count + class_counts.max()*0.01, f"{count} ({pct}%)", ha='center', va='bottom')
        ax.set_title('Class Distribution (Imbalanced)')
        ax.set_ylabel('Count')
        ax.set_xlabel('Class')
        
        ax = axes[0, 1]
        for cls, color in [('Normal', 'green'), ('Fraud', 'red')]:
            class_data = df[df['Class'] == (0 if cls == 'Normal' else 1)]['Amount']
            sns.histplot(class_data[class_data <= df['Amount'].quantile(0.99)], bins=60, kde=True, stat='density', element='step', fill=False, label=cls, color=color, ax=ax)
        ax.set_xlabel('Amount (trimmed 99% quantile)')
        ax.set_ylabel('Density')
        ax.set_title('Amount Density by Class')
        ax.legend()

        ax = axes[0, 2]
        df['Time_hour'] = (df['Time'] // 3600) % 24
        time_by_class = df.groupby(['Time_hour', 'Class']).size().unstack(fill_value=0)
        time_by_class_norm = time_by_class.div(time_by_class.sum(axis=0), axis=1)*100
        time_by_class_norm.plot(ax=ax, marker='o', color=['green', 'red'])
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Percent of Transactions (%)')
        ax.set_title('Hourly Transaction Share by Class')
        ax.legend(['Normal', 'Fraud'])

        ax = axes[1, 0]
        feature_cols = [c for c in df.columns if c not in ['Class', 'Time', 'Amount']]
        cols_for_corr = feature_cols + ['Class']
        corr = df[cols_for_corr].corr()['Class'].drop('Class').abs().sort_values(ascending=True)
        corr.plot(kind='barh', ax=ax, color='teal')
        ax.set_title('Absolute Correlation with Class')
        ax.set_xlabel('Pearson |r|')

        ax = axes[1, 1]
        v_cols = [c for c in df.columns if c.startswith('V')]
        if len(v_cols) > 0:
            cols_for_vcorr = v_cols + ['Class']
            v_corr = df[cols_for_vcorr].corr()['Class'].drop('Class').abs().sort_values(ascending=False).head(10)
            sns.barplot(x=v_corr.values, y=v_corr.index, palette='viridis', ax=ax)
            ax.set_title('Top 10 V-features Abs Correlation')
            ax.set_xlabel('Absolute Correlation')
            ax.set_ylabel('Feature')
        else:
            ax.text(0.5, 0.5, 'No V-features available', ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('Top 10 V-features Abs Correlation')

        ax = axes[1, 2]
        df_plot = df[df['Amount'] < df['Amount'].quantile(0.99)].copy()
        df_plot['Class_Label'] = df_plot['Class'].map({0: 'Normal', 1: 'Fraud'})
        sns.boxplot(x='Class_Label', y='Amount', data=df_plot, palette=palette, ax=ax)
        ax.set_title('Amount Boxplot by Class (trimmed)')
        ax.set_xlabel('Class')
        ax.set_ylabel('Amount')

        fig.tight_layout()
        fig.savefig(out_dir / 'eda_creditcard.png', dpi=120)
        plt.close(fig)

        if 'Time_hour' in df.columns:
            df.drop('Time_hour', axis=1, inplace=True)

        CreditCardEDA._logger.info("CreditCard EDA saved to eda_creditcard.png")
