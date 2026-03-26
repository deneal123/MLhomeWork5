"""
EDA для BankSim датасета (используется в Task1 и Task4)
Требуется минимум 4 графика
"""
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

from antifraud.utils.logger import get_logger


class BankSimEDA:
    _logger = None

    @staticmethod
    def run(df, out_dir: Path):
        if BankSimEDA._logger is None:
            BankSimEDA._logger = get_logger('BankSimEDA')
        
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Распределение классов (fraud vs normal)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        ax = axes[0, 0]
        fraud_counts = df['fraud'].value_counts()
        ax.pie(fraud_counts, labels=['Normal', 'Fraud'], autopct='%1.1f%%', 
               colors=['green', 'red'], explode=[0, 0.1])
        ax.set_title('Class Distribution (Fraud vs Normal)')
        
        # 2. Распределение сумм транзакций
        ax = axes[0, 1]
        df[df['fraud'] == 0]['amount'].hist(bins=50, ax=ax, alpha=0.6, label='Normal', color='green')
        df[df['fraud'] == 1]['amount'].hist(bins=50, ax=ax, alpha=0.6, label='Fraud', color='red')
        ax.set_xlabel('Amount')
        ax.set_ylabel('Frequency')
        ax.set_title('Amount Distribution by Class')
        ax.legend()
        ax.set_xlim(0, df['amount'].quantile(0.99))
        
        # 3. Категории транзакций
        ax = axes[0, 2]
        cat_fraud = df[df['fraud'] == 1]['category'].value_counts().head(8)
        cat_fraud.plot(kind='barh', ax=ax, color='red')
        ax.set_title('Top Fraud Categories')
        ax.set_xlabel('Count')
        
        # 4. Возрастное распределение
        ax = axes[1, 0]
        df['age'].value_counts().sort_index().plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title('Age Distribution')
        ax.set_xlabel('Age Group')
        ax.set_ylabel('Count')
        
        # 5. Gender распределение
        ax = axes[1, 1]
        gender_fraud = df.groupby(['gender', 'fraud']).size().unstack()
        gender_fraud.plot(kind='bar', ax=ax, color=['green', 'red'])
        ax.set_title('Gender vs Fraud')
        ax.set_xlabel('Gender')
        ax.set_ylabel('Count')
        ax.legend(['Normal', 'Fraud'])
        
        # 6. Корреляция признаков с fraud
        ax = axes[1, 2]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_with_fraud = df[numeric_cols].corr()['fraud'].drop('fraud').abs().sort_values(ascending=True)
        corr_with_fraud.plot(kind='barh', ax=ax, color='teal')
        ax.set_title('Feature Correlation with Fraud')
        
        fig.tight_layout()
        fig.savefig(out_dir / 'eda_banksim.png', dpi=100)
        plt.close(fig)

        BankSimEDA._logger.info("BankSim EDA saved to eda_banksim.png")
