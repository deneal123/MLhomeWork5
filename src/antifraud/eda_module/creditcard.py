"""
EDA для CreditCard датасета (используется в Task2)
Требуется минимум 4 графика
"""
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import pandas as pd

from antifraud.utils.logger import get_logger


class CreditCardEDA:
    _logger = None

    @staticmethod
    def run(df, out_dir: Path):
        if CreditCardEDA._logger is None:
            CreditCardEDA._logger = get_logger('CreditCardEDA')
        
        # Создаём копию, чтобы не модифицировать оригинальный DataFrame
        df = df.copy()
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Проверяем наличие необходимых колонок
        required_cols = ['Class', 'Time', 'Amount']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            CreditCardEDA._logger.warning(f"Missing columns: {missing}. Available: {df.columns.tolist()}")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Распределение классов
        ax = axes[0, 0]
        class_counts = df['Class'].value_counts()
        ax.pie(class_counts, labels=['Normal', 'Fraud'], autopct='%1.3f%%',
               colors=['green', 'red'], explode=[0, 0.1])
        ax.set_title('Class Distribution (Imbalanced)')
        
        # 2. Распределение сумм транзакций
        ax = axes[0, 1]
        df[df['Class'] == 0]['Amount'].hist(bins=50, ax=ax, alpha=0.6, label='Normal', color='green')
        df[df['Class'] == 1]['Amount'].hist(bins=50, ax=ax, alpha=0.6, label='Fraud', color='red')
        ax.set_xlabel('Amount')
        ax.set_ylabel('Frequency')
        ax.set_title('Amount Distribution by Class')
        ax.legend()
        ax.set_xlim(0, df['Amount'].quantile(0.99))
        
        # 3. Временное распределение
        ax = axes[0, 2]
        df['Time_hour'] = (df['Time'] // 3600) % 24
        fraud_time = df[df['Class'] == 1]['Time_hour']
        normal_time = df[df['Class'] == 0]['Time_hour']
        ax.hist([normal_time, fraud_time], bins=24, alpha=0.6, label=['Normal', 'Fraud'], color=['green', 'red'])
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Frequency')
        ax.set_title('Transaction Time Distribution')
        ax.legend()
        
        # 4. Корреляция признаков с Class
        ax = axes[1, 0]
        # Для расчёта корреляции нужно включить Class
        feature_cols = [c for c in df.columns if c not in ['Class', 'Time', 'Amount']]
        # Добавляем Class временно для расчёта корреляции
        cols_for_corr = feature_cols + ['Class']
        corr = df[cols_for_corr].corr()['Class'].drop('Class').abs().sort_values(ascending=True)
        corr.plot(kind='barh', ax=ax, color='teal')
        ax.set_title('Feature Correlation with Class')
        
        # 5. Топ признаки V (V1-V28)
        ax = axes[1, 1]
        v_cols = [c for c in df.columns if c.startswith('V')]
        cols_for_vcorr = v_cols + ['Class']
        v_corr = df[cols_for_vcorr].corr()['Class'].drop('Class').abs().sort_values(ascending=False).head(10)
        v_corr.plot(kind='bar', ax=ax, color='purple')
        ax.set_title('Top 10 V-features Correlation')
        ax.set_ylabel('Absolute Correlation')
        
        # 6. Box plot для Amount по классам
        ax = axes[1, 2]
        df_plot = df[df['Amount'] < df['Amount'].quantile(0.99)].copy()
        df_plot['Class_Label'] = df_plot['Class'].map({0: 'Normal', 1: 'Fraud'})
        df_plot.boxplot(column='Amount', by='Class_Label', ax=ax)
        ax.set_title('Amount Boxplot by Class')
        ax.set_xlabel('Class')
        ax.set_ylabel('Amount')
        
        fig.tight_layout()
        fig.savefig(out_dir / 'eda_creditcard.png', dpi=100)
        plt.close(fig)

        # Time_hour создаётся в этой функции, удаляем только если создали
        if 'Time_hour' in df.columns:
            df.drop('Time_hour', axis=1, inplace=True)
        
        CreditCardEDA._logger.info("CreditCard EDA saved to eda_creditcard.png")
