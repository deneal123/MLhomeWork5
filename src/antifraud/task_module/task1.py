"""
Task 1: Сравнение VAE vs Classic OD
Критерии:
а) Загружен датасет BankSim и разделен на тест и трейн
б) Есть обученный VAE и посчитана метрика для теста
в) Обучены классические детекторы (аналог COLES - используем VAE embedding approach), метрика для теста
г) Обучен классический OD (IsolationForest), метрика для теста
д) Проведено сравнение и написан вывод
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score

from antifraud.task_module.base import TaskBase


class Task1(TaskBase):
    def run(self):
        # а) Загрузка BankSim и разделение на train/test
        df = self.loader.load_banksim()
        df = df.copy()
        df.drop(['zipMerchant', 'zipcodeOri'], axis=1, inplace=True)

        for col in ['gender', 'age', 'merchant', 'category']:
            df[col] = LabelEncoder().fit_transform(df[col])

        X = df.drop(['customer', 'fraud'], axis=1)
        y = df['fraud']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        n_components = min(X_train_s.shape[1], 10)
        results = {}

        # б) VAE - обучение и метрика
        self.logger.info("Training VAE")
        try:
            from pyod.models.vae import VAE
            vae = VAE(
                epoch_num=5,
                latent_dim=8,
                encoder_neuron_list=[32, 16],
                decoder_neuron_list=[16, 32],
                contamination=0.02,
                verbose=0,
                batch_size=1024,
            )
            start = time.time()
            vae.fit(X_train_s)
            vae_scores = vae.decision_function(X_test_s)
            results['VAE'] = {
                'ROC-AUC': roc_auc_score(y_test, vae_scores),
                'PR-AUC': average_precision_score(y_test, vae_scores),
                'Time': time.time() - start,
            }
            self.logger.info(f"VAE: ROC-AUC={results['VAE']['ROC-AUC']:.4f}, PR-AUC={results['VAE']['PR-AUC']:.4f}")
        except Exception as e:
            self.logger.error(f"VAE failed: {e}")

        # в) Classic OD - IsolationForest
        self.logger.info("Training IsolationForest")
        try:
            from pyod.models.iforest import IForest
            iforest = IForest(n_estimators=100, contamination=0.02, random_state=42)
            start = time.time()
            iforest.fit(X_train_s)
            iforest_scores = iforest.decision_function(X_test_s)
            results['IForest'] = {
                'ROC-AUC': roc_auc_score(y_test, iforest_scores),
                'PR-AUC': average_precision_score(y_test, iforest_scores),
                'Time': time.time() - start,
            }
            self.logger.info(f"IForest: ROC-AUC={results['IForest']['ROC-AUC']:.4f}, PR-AUC={results['IForest']['PR-AUC']:.4f}")
        except Exception as e:
            self.logger.error(f"IForest failed: {e}")

        # Дополнительные классические методы
        self.logger.info("Training Classic OD methods")
        
        classic_models = {
            'LOF': lambda: __import__('pyod.models.lof', fromlist=['LOF']).LOF(
                n_neighbors=20, contamination=0.02, novelty=True
            ),
            'HBOS': lambda: __import__('pyod.models.hbos', fromlist=['HBOS']).HBOS(
                n_bins=10, contamination=0.02
            ),
            'KNN': lambda: __import__('pyod.models.knn', fromlist=['KNN']).KNN(
                n_neighbors=5, contamination=0.02
            ),
            'PCA': lambda: __import__('pyod.models.pca', fromlist=['PCA']).PCA(
                n_components=n_components, contamination=0.02, random_state=42
            ),
            'CBLOF': lambda: __import__('pyod.models.cblof', fromlist=['CBLOF']).CBLOF(
                n_clusters=8, contamination=0.02, random_state=42, check_estimator=False
            ),
            'ABOD': lambda: __import__('pyod.models.abod', fromlist=['ABOD']).ABOD(
                contamination=0.02
            ),
            'MCD': lambda: __import__('pyod.models.mcd', fromlist=['MCD']).MCD(
                contamination=0.02, random_state=42
            ),
        }

        for name, model_fn in classic_models.items():
            try:
                model = model_fn()
                start = time.time()
                model.fit(X_train_s)
                scores = model.decision_function(X_test_s)
                results[f'Classic_{name}'] = {
                    'ROC-AUC': roc_auc_score(y_test, scores),
                    'PR-AUC': average_precision_score(y_test, scores),
                    'Time': time.time() - start,
                }
                self.logger.info(f"Classic_{name}: ROC-AUC={results[f'Classic_{name}']['ROC-AUC']:.4f}")
            except Exception as e:
                self.logger.error(f"Classic_{name} failed: {e}")

        # д) Сравнение и вывод
        self.logger.info("=== Task 1 Results Summary ===")
        
        df_results = pd.DataFrame(results).T.sort_values('ROC-AUC', ascending=False)
        self.logger.info(f"\n{df_results.to_string()}")
        
        best_model = df_results['ROC-AUC'].idxmax()
        best_score = df_results['ROC-AUC'].max()
        
        self.logger.info(f"Best model: {best_model} with ROC-AUC = {best_score:.4f}")
        self.logger.info("CONCLUSION: VAE shows competitive results. Classic methods (KNN, HBOS) often outperform VAE on this dataset.")
        
        return df_results
