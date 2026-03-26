"""
Antifraud Pipeline - Main application
Запускает все 4 задания с EDA анализом
"""
import pandas as pd
import matplotlib.pyplot as plt
import torch

from antifraud.data_module import DataLoader
from antifraud.eda_module import BankSimEDA, CreditCardEDA, WeiboEDA
from antifraud.task_module import Task1, Task2, Task3, Task4
from antifraud.utils.logger import get_logger


class AntifraudPipeline:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.loader = DataLoader()
        self.task1 = Task1(self.loader)
        self.task2 = Task2(self.loader)
        self.task3 = Task3(self.loader)
        self.task4 = Task4(self.loader)

    def _check_gpu(self):
        if torch.cuda.is_available():
            self.logger.info('GPU доступен')
        else:
            self.logger.info('GPU не доступен')

    def run(self):
        self._check_gpu()

        print("\n" + "="*60)
        print("EDA Analysis")
        print("="*60)
        
        # EDA для BankSim (Task1, Task4)
        print("\n--- BankSim EDA ---")
        bank_df = self.loader.load_banksim()
        BankSimEDA.run(bank_df, self.loader.output_root)

        # EDA для CreditCard (Task2)
        print("\n--- CreditCard EDA ---")
        credit_df = self.loader.load_creditcard()
        CreditCardEDA.run(credit_df, self.loader.output_root)

        # EDA для Weibo (Task3)
        print("\n--- Weibo EDA ---")
        weibo_data = self.loader.load_weibo()
        if weibo_data is not None:
            WeiboEDA.run(weibo_data, self.loader.output_root)
        else:
            self.logger.warning('Weibo недоступен')
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, 'Weibo не загружен', ha='center', va='center', fontsize=12)
            ax.axis('off')
            fig.savefig(self.loader.output_path('eda_weibo_missing.png'), dpi=100)
            plt.close(fig)

        # Запуск заданий
        print("\n" + "="*60)
        print("Task 1: VAE vs AutoEncoder vs Classic OD")
        print("="*60)
        results1 = self.task1.run()
        results1.to_csv(self.loader.output_path('task1_results.csv'))
        
        # График результатов Task1
        fig, ax = plt.subplots(figsize=(10, 6))
        results1['ROC-AUC'].sort_values().plot(kind='barh', ax=ax, color='steelblue')
        ax.set_xlabel('ROC-AUC')
        ax.set_title('Task 1: VAE vs AE vs Classic OD')
        ax.axvline(x=0.5, color='red', linestyle='--', label='Random')
        fig.tight_layout()
        fig.savefig(self.loader.output_path('task1_roc_auc.png'), dpi=100)
        plt.close(fig)

        print("\n" + "="*60)
        print("Task 2: Unsupervised Detectors on CreditCard")
        print("="*60)
        results2 = self.task2.run()
        results2.to_csv(self.loader.output_path('task2_results.csv'))

        # График результатов Task2
        fig, ax = plt.subplots(figsize=(12, 6))
        results2['ROC-AUC'].sort_values().plot(kind='barh', ax=ax, color='teal')
        ax.set_xlabel('ROC-AUC')
        ax.set_title('Task 2: Unsupervised Detectors Comparison')
        ax.axvline(x=0.5, color='red', linestyle='--', label='Random')
        fig.tight_layout()
        fig.savefig(self.loader.output_path('task2_roc_auc.png'), dpi=100)
        plt.close(fig)

        print("\n" + "="*60)
        print("Task 3: Weibo Anomaly Detection")
        print("="*60)
        results3 = self.task3.run(weibo_data)
        if isinstance(results3, pd.DataFrame):
            results3.to_csv(self.loader.output_path('task3_results.csv'))
            
            # График результатов Task3
            fig, ax = plt.subplots(figsize=(8, 5))
            results3.set_index('Model')['ROC-AUC'].sort_values().plot(kind='barh', ax=ax, color='purple')
            ax.set_xlabel('ROC-AUC')
            ax.set_title('Task 3: Weibo Anomaly Detection')
            ax.axvline(x=0.9, color='green', linestyle='--', label='Target 0.9')
            ax.axvline(x=0.5, color='red', linestyle='--', label='Random')
            ax.legend()
            fig.tight_layout()
            fig.savefig(self.loader.output_path('task3_roc_auc.png'), dpi=100)
            plt.close(fig)
        else:
            pd.DataFrame(results3).to_csv(self.loader.output_path('task3_results.csv'))

        print("\n" + "="*60)
        print("Task 4: PYOD vs PYTOD Comparison")
        print("="*60)
        results4 = self.task4.run()
        results4.to_csv(self.loader.output_path('task4_results.csv'))
        
        # График результатов Task4
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['steelblue' if lib == 'PYOD' else 'orange' for lib in results4['Library']]
        results4.sort_values('ROC-AUC').plot(kind='barh', x='Model', y='ROC-AUC', ax=ax, color=colors)
        ax.set_xlabel('ROC-AUC')
        ax.set_title('Task 4: PYOD vs PYTOD')
        ax.axvline(x=0.5, color='red', linestyle='--', label='Random')
        
        # Легенда для библиотек
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='steelblue', label='PYOD'),
                          Patch(facecolor='orange', label='PYTOD')]
        ax.legend(handles=legend_elements)
        
        fig.tight_layout()
        fig.savefig(self.loader.output_path('task4_roc_auc.png'), dpi=100)
        plt.close(fig)
        
        print("\n" + "="*60)
        print("All tasks completed! Results saved to outputs/")
        print("="*60)


def main():
    AntifraudPipeline().run()
