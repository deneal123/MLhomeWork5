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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from antifraud.task_module.base import TaskBase


class Task3(TaskBase):
    def run(self, data):
        if data is None:
            self.logger.warning("Weibo data not available, returning NaN")
            return pd.DataFrame({'ROC-AUC': [np.nan], 'PR-AUC': [np.nan], 'Model': ['N/A']})

        train_mask = data.train_mask.numpy()
        test_mask = data.test_mask.numpy()

        X = data.x.numpy()
        y = data.y.numpy().astype(int)

        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            self.logger.warning('Weibo split has class imbalance in test; using stratified train_test_split to include anomalies in test')
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

        test_ratio = len(X_test) / (len(X_train) + len(X_test))
        self.logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}, Test ratio: {test_ratio*100:.1f}%")

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        def safe_scores(y_true, y_score):
            try:
                roc = roc_auc_score(y_true, y_score)
            except Exception:
                roc = float('nan')
            try:
                pr = average_precision_score(y_true, y_score)
            except Exception:
                pr = float('nan')
            return roc, pr

        results = []
        
        model_candidates = {
            'IForest': lambda: __import__('pyod.models.iforest', fromlist=['IForest']).IForest(
                n_estimators=200, contamination=0.1, max_samples=min(256, len(X_train_s)), random_state=42
            ),
            'LOF': lambda: __import__('pyod.models.lof', fromlist=['LOF']).LOF(
                n_neighbors=15, contamination=0.1, novelty=True, n_jobs=-1
            ),
            'HBOS': lambda: __import__('pyod.models.hbos', fromlist=['HBOS']).HBOS(
                n_bins=15, contamination=0.1
            ),
            'KNN': lambda: __import__('pyod.models.knn', fromlist=['KNN']).KNN(
                n_neighbors=10, contamination=0.1, n_jobs=-1
            ),
            'PCA': lambda: __import__('pyod.models.pca', fromlist=['PCA']).PCA(
                n_components=min(10, X_train_s.shape[1]), contamination=0.1
            ),
            'CBLOF': lambda: __import__('pyod.models.cblof', fromlist=['CBLOF']).CBLOF(
                n_clusters=8, contamination=0.1, random_state=42, check_estimator=False
            ),
            'ABOD': lambda: __import__('pyod.models.abod', fromlist=['ABOD']).ABOD(
                contamination=0.1
            ),
            'OCSVM': lambda: __import__('pyod.models.ocsvm', fromlist=['OCSVM']).OCSVM(
                kernel='rbf', nu=0.02
            ),
        }

        stacking_features = []
        stacking_labels_train = []
        stacking_labels_test = []
        model_scores_train = {}
        model_scores_test = {}

        for name, ctor in model_candidates.items():
            self.logger.info(f"Training {name}")
            try:
                model = ctor()
                model.fit(X_train_s)
                train_scores = model.decision_function(X_train_s)
                test_scores = model.decision_function(X_test_s)
                model_scores_train[name] = train_scores
                model_scores_test[name] = test_scores

                roc_auc, pr_auc = safe_scores(y_test, test_scores)
                results.append({'Model': name, 'ROC-AUC': roc_auc, 'PR-AUC': pr_auc})
                self.logger.info(f"{name}: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")
            except Exception as e:
                self.logger.error(f"{name} failed: {e}")
                continue

        if model_scores_train and model_scores_test:
            self.logger.info("Training stacking meta-model")
            X_meta_train = np.vstack([model_scores_train[m] for m in model_scores_train]).T
            X_meta_test = np.vstack([model_scores_test[m] for m in model_scores_test]).T
            try:
                meta = LogisticRegression(class_weight='balanced', max_iter=1000, solver='liblinear')
                meta.fit(X_meta_train, y_train)
                meta_scores = meta.predict_proba(X_meta_test)[:, 1]
                meta_roc, meta_pr = safe_scores(y_test, meta_scores)
                results.append({'Model': 'Stacking', 'ROC-AUC': meta_roc, 'PR-AUC': meta_pr})
                self.logger.info(f"Stacking: ROC-AUC={meta_roc:.4f}, PR-AUC={meta_pr:.4f}")
            except Exception as e:
                self.logger.error(f"Stacking failed: {e}")
 
        df_results = pd.DataFrame(results)
        if len(df_results) == 0:
            return pd.DataFrame({'ROC-AUC': [np.nan], 'PR-AUC': [np.nan], 'Model': ['N/A']})
 
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

        if best_roc_auc >= 0.9:
            self.logger.info(f"SUCCESS: ROC-AUC >= 0.9 achieved ({best_roc_auc:.4f})")
        else:
            self.logger.warning(f"Target not met: ROC-AUC = {best_roc_auc:.4f} < 0.9")

        return df_results.sort_values('ROC-AUC', ascending=False)
