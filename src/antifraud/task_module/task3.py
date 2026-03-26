"""
Task3: Разбор на предмет аномалий датасета Weibo
Критерии:
а) Загружен датасет Weibo и разделен на тест (≥20%) и трейн
б) Использован один из алгоритмов из семинара
в) Есть обученный алгоритм и посчитана метрика на тесте
г) Получен ROC AUC не меньше 0.9 на тесте
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

from antifraud.task_module.base import TaskBase


class Task3(TaskBase):
    def run(self, data):
        if data is None:
            self.logger.warning("Weibo data not available, returning NaN")
            return pd.DataFrame({'ROC-AUC': [np.nan], 'PR-AUC': [np.nan], 'Model': ['N/A']})

        # а) Разделение на train/test (тест ≥20%)
        train_mask = data.train_mask.numpy()
        test_mask = data.test_mask.numpy()

        X = data.x.numpy()
        y = data.y.numpy().astype(int)

        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]

        test_ratio = len(X_test) / (len(X_train) + len(X_test))
        self.logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}, Test ratio: {test_ratio*100:.1f}%")

        # Стандартизация
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        results = []

        # б,в) Используем несколько алгоритмов и выбираем лучший
        
        # 1. IsolationForest
        self.logger.info("Training IForest")
        try:
            from pyod.models.iforest import IForest
            model = IForest(
                n_estimators=200,
                contamination=0.1,
                max_samples=min(256, len(X_train_s)),
                random_state=42
            )
            model.fit(X_train_s)
            scores = model.decision_function(X_test_s)
            roc_auc = roc_auc_score(y_test, scores)
            pr_auc = average_precision_score(y_test, scores)
            results.append({
                'Model': 'IForest',
                'ROC-AUC': roc_auc,
                'PR-AUC': pr_auc,
            })
            self.logger.info(f"IForest: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")
        except Exception as e:
            self.logger.error(f"IForest failed: {e}")

        # 2. LOF
        self.logger.info("Training LOF")
        try:
            from pyod.models.lof import LOF
            lof = LOF(n_neighbors=15, contamination=0.1, novelty=True, n_jobs=-1)
            lof.fit(X_train_s)
            scores_lof = lof.decision_function(X_test_s)
            roc_auc = roc_auc_score(y_test, scores_lof)
            pr_auc = average_precision_score(y_test, scores_lof)
            results.append({
                'Model': 'LOF',
                'ROC-AUC': roc_auc,
                'PR-AUC': pr_auc,
            })
            self.logger.info(f"LOF: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")
        except Exception as e:
            self.logger.error(f"LOF failed: {e}")

        # 3. HBOS
        self.logger.info("Training HBOS")
        try:
            from pyod.models.hbos import HBOS
            hbos = HBOS(n_bins=15, contamination=0.1)
            hbos.fit(X_train_s)
            scores_hbos = hbos.decision_function(X_test_s)
            roc_auc = roc_auc_score(y_test, scores_hbos)
            pr_auc = average_precision_score(y_test, scores_hbos)
            results.append({
                'Model': 'HBOS',
                'ROC-AUC': roc_auc,
                'PR-AUC': pr_auc,
            })
            self.logger.info(f"HBOS: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")
        except Exception as e:
            self.logger.error(f"HBOS failed: {e}")

        # 4. KNN
        self.logger.info("Training KNN")
        try:
            from pyod.models.knn import KNN
            knn = KNN(n_neighbors=10, contamination=0.1, n_jobs=-1)
            knn.fit(X_train_s)
            scores_knn = knn.decision_function(X_test_s)
            roc_auc = roc_auc_score(y_test, scores_knn)
            pr_auc = average_precision_score(y_test, scores_knn)
            results.append({
                'Model': 'KNN',
                'ROC-AUC': roc_auc,
                'PR-AUC': pr_auc,
            })
            self.logger.info(f"KNN: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")
        except Exception as e:
            self.logger.error(f"KNN failed: {e}")

        # 5. PCA
        self.logger.info("Training PCA")
        try:
            from pyod.models.pca import PCA
            pca = PCA(n_components=min(10, X_train_s.shape[1]), contamination=0.1, random_state=42)
            pca.fit(X_train_s)
            scores_pca = pca.decision_function(X_test_s)
            roc_auc = roc_auc_score(y_test, scores_pca)
            pr_auc = average_precision_score(y_test, scores_pca)
            results.append({
                'Model': 'PCA',
                'ROC-AUC': roc_auc,
                'PR-AUC': pr_auc,
            })
            self.logger.info(f"PCA: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")
        except Exception as e:
            self.logger.error(f"PCA failed: {e}")

        # Создаём DataFrame и находим лучшую модель
        df_results = pd.DataFrame(results)
        if len(df_results) == 0:
            return pd.DataFrame({'ROC-AUC': [np.nan], 'PR-AUC': [np.nan], 'Model': ['N/A']})

        # Если ROC-AUC все NaN (например, один класс в y_test), то idxmax упадёт
        if df_results['ROC-AUC'].isna().all():
            best_roc_auc = float('nan')
            best_model = 'N/A'
        else:
            best_idx = df_results['ROC-AUC'].idxmax()
            best_roc_auc = df_results.loc[best_idx, 'ROC-AUC']
            best_model = df_results.loc[best_idx, 'Model']

        self.logger.info("=== Task3 Results ===")
        self.logger.info(df_results.to_string(index=False))
        self.logger.info(f"Best model: {best_model} with ROC-AUC = {best_roc_auc:.4f}")

        # г) Проверка достижения ROC-AUC ≥ 0.9
        if best_roc_auc >= 0.9:
            self.logger.info(f"SUCCESS: ROC-AUC >= 0.9 achieved ({best_roc_auc:.4f})")
        else:
            self.logger.warning(f"Target not met: ROC-AUC = {best_roc_auc:.4f} < 0.9")

        return df_results.sort_values('ROC-AUC', ascending=False)
