"""
Task2: Сравнение качества работы unsupervised детекторов аномалий на CreditCard
Критерии:
а) Загружен датасет CreditCard и разделен на тест (≥20%) и трейн
б) Есть обученные unsupervised детекторы аномалий (≥6) и посчитаны метрики для теста
в) Сделать вывод метрики для всех алгоритмов в единую табличку
г) Проведено сравнение и написан вывод
"""
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

from antifraud.task_module.base import TaskBase


class Task2(TaskBase):
    def run(self):
        df = self.loader.load_creditcard()
        X = df.drop(['Class', 'Time'], axis=1)
        y = df['Class']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        self.logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        self.logger.info(f"Test ratio: {len(X_test)/len(X)*100:.1f}%")

        detectors = {
            'IForest': __import__('pyod.models.iforest', fromlist=['IForest']).IForest(
                n_estimators=100, contamination=0.0017, random_state=42
            ),
            'LOF': __import__('pyod.models.lof', fromlist=['LOF']).LOF(
                n_neighbors=20, contamination=0.0017
            ),
            'HBOS': __import__('pyod.models.hbos', fromlist=['HBOS']).HBOS(
                n_bins=10, contamination=0.0017
            ),
            'KNN': __import__('pyod.models.knn', fromlist=['KNN']).KNN(
                n_neighbors=5, contamination=0.0017
            ),
            'PCA': __import__('pyod.models.pca', fromlist=['PCA']).PCA(
                n_components=5, contamination=0.0017, random_state=42
            ),
            # Дополнительные методы
            'CBLOF': __import__('pyod.models.cblof', fromlist=['CBLOF']).CBLOF(
                n_clusters=8, contamination=0.0017, random_state=42, check_estimator=False
            ),
            'ABOD': __import__('pyod.models.abod', fromlist=['ABOD']).ABOD(
                contamination=0.0017
            ),
            'LODA': __import__('pyod.models.loda', fromlist=['LODA']).LODA(
                n_bins=10, contamination=0.0017
            ),
            'MCD': __import__('pyod.models.mcd', fromlist=['MCD']).MCD(
                contamination=0.0017, random_state=42
            ),
            'OCSVM': __import__('pyod.models.ocsvm', fromlist=['OCSVM']).OCSVM(
                kernel='rbf', nu=0.0017, contamination=0.0017
            ),
            'SOS': __import__('pyod.models.sos', fromlist=['SOS']).SOS(
                contamination=0.0017
            ),
        }

        results = {}
        for name, model in tqdm(detectors.items(), desc='Task2 detectors', unit='model'):
            try:
                start = time.time()
                model.fit(X_train_s)
                scores = model.decision_function(X_test_s)
                results[name] = {
                    'ROC-AUC': roc_auc_score(y_test, scores),
                    'PR-AUC': average_precision_score(y_test, scores),
                    'Time': time.time() - start,
                }
                self.logger.info(f"{name}: ROC-AUC={results[name]['ROC-AUC']:.4f}, PR-AUC={results[name]['PR-AUC']:.4f}")
            except Exception as e:
                self.logger.error(f"{name} failed: {e}")

        df_results = pd.DataFrame(results).T.sort_values('ROC-AUC', ascending=False)
        
        self.logger.info("=== Task2 Results Summary ===")
        self.logger.info(df_results.to_string())
        
        best_model = df_results['ROC-AUC'].idxmax()
        best_score = df_results['ROC-AUC'].max()
        self.logger.info(f"Best model: {best_model} with ROC-AUC = {best_score:.4f}")
        
        mean_auc = df_results['ROC-AUC'].mean()
        std_auc = df_results['ROC-AUC'].std()
        self.logger.info(f"Mean ROC-AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        
        return df_results
