"""
Task4: Сравнение PYOD vs PYTOD (10 алгоритмов)
Критерии:
а) Загружен датасет BankSim и разделен на тест
б) Есть обученные алгоритмы из PYOD, посчитана скорость и метрика для теста
в) Есть обученные аналогичные алгоритмы из PYTOD, посчитана скорость и метрика для теста
г) Собрать скорость и метрики для всех алгоритмов в единую табличку
д) Проведено сравнение и написан вывод
*) Для заданий со словом "сравнить" нужна статзначимость
"""
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score

from antifraud.task_module.base import TaskBase


class Task4(TaskBase):
    def run(self):
        # а) Загрузка BankSim и разделение
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

        results = []

        # б) PYOD модели (10 алгоритмов)
        pyod_models = {
            'IForest': lambda: __import__('pyod.models.iforest', fromlist=['IForest']).IForest(
                n_estimators=100, contamination=0.02, random_state=42
            ),
            'LOF': lambda: __import__('pyod.models.lof', fromlist=['LOF']).LOF(
                n_neighbors=20, contamination=0.02
            ),
            'HBOS': lambda: __import__('pyod.models.hbos', fromlist=['HBOS']).HBOS(
                n_bins=10, contamination=0.02
            ),
            'KNN': lambda: __import__('pyod.models.knn', fromlist=['KNN']).KNN(
                n_neighbors=5, contamination=0.02
            ),
            'PCA': lambda: __import__('pyod.models.pca', fromlist=['PCA']).PCA(
                n_components=5, contamination=0.02, random_state=42
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
            'LODA': lambda: __import__('pyod.models.loda', fromlist=['LODA']).LODA(
                n_bins=10, contamination=0.02
            ),
            'OCSVM': lambda: __import__('pyod.models.ocsvm', fromlist=['OCSVM']).OCSVM(
                kernel='rbf', nu=0.02, contamination=0.02
            ),
        }

        self.logger.info("Training PYOD models")
        pyod_scores = {}
        for name, model_fn in tqdm(pyod_models.items(), desc='PYOD', unit='model'):
            try:
                model = model_fn()
                start = time.time()
                model.fit(X_train_s)
                train_time = time.time() - start
                
                scores = model.decision_function(X_test_s)
                pyod_scores[name] = scores
                
                results.append({
                    'Library': 'PYOD',
                    'Model': name,
                    'ROC-AUC': roc_auc_score(y_test, scores),
                    'PR-AUC': average_precision_score(y_test, scores),
                    'Train Time (s)': train_time,
                })
                self.logger.info(f"PYOD {name}: ROC-AUC={roc_auc_score(y_test, scores):.4f}, Time={train_time:.2f}s")
            except Exception as e:
                self.logger.error(f"PYOD {name} failed: {e}")

        # в) PYTOD модели (доступно 5 моделей)
        pytod_models = {
            'IForest': lambda: __import__('pytod.models.iforest', fromlist=['IForest']).IForest(
                n_estimators=100, contamination=0.02, random_state=42
            ),
            'LOF': lambda: __import__('pytod.models.lof', fromlist=['LOF']).LOF(
                n_neighbors=20, contamination=0.02
            ),
            'HBOS': lambda: __import__('pytod.models.hbos', fromlist=['HBOS']).HBOS(
                n_bins=10, contamination=0.02
            ),
            'KNN': lambda: __import__('pytod.models.knn', fromlist=['KNN']).KNN(
                n_neighbors=5, contamination=0.02
            ),
            'PCA': lambda: __import__('pytod.models.pca', fromlist=['PCA']).PCA(
                n_components=5, contamination=0.02, random_state=42
            ),
        }

        self.logger.info("Training PYTOD models")
        pytod_scores = {}
        for name, model_fn in tqdm(pytod_models.items(), desc='PYTOD', unit='model'):
            try:
                model = model_fn()
                start = time.time()
                model.fit(X_train_s)
                train_time = time.time() - start
                
                scores = model.decision_function(X_test_s)
                pytod_scores[name] = scores
                
                results.append({
                    'Library': 'PYTOD',
                    'Model': name,
                    'ROC-AUC': roc_auc_score(y_test, scores),
                    'PR-AUC': average_precision_score(y_test, scores),
                    'Train Time (s)': train_time,
                })
                self.logger.info(f"PYTOD {name}: ROC-AUC={roc_auc_score(y_test, scores):.4f}, Time={train_time:.2f}s")
            except Exception as e:
                self.logger.error(f"PYTOD {name} failed: {e}")

        # г) Единая таблица
        df_results = pd.DataFrame(results)
        
        # д) Сравнение и статистическая значимость
        self.logger.info("=== Task4 Results Summary ===")
        self.logger.info(df_results.to_string(index=False))

        # Статистический анализ для пар с одинаковыми моделями
        self.logger.info("=== Statistical Significance (paired t-test) ===")
 
        significance_results = []
        for model_name in pytod_scores.keys():
            if model_name in pyod_scores:
                pyod_auc = roc_auc_score(y_test, pyod_scores[model_name])
                pytod_auc = roc_auc_score(y_test, pytod_scores[model_name])
 
                # Bootstrap для оценки стабильности
                n_bootstrap = 100
                pyod_bootstrap = []
                pytod_bootstrap = []
 
                for _ in range(n_bootstrap):
                    idx = np.random.choice(len(X_test_s), size=len(X_test_s), replace=True)
                    y_bs = y_test.iloc[idx] if hasattr(y_test, 'iloc') else y_test[idx]
 
                    pyod_bootstrap.append(roc_auc_score(y_bs, pyod_scores[model_name]))
                    pytod_bootstrap.append(roc_auc_score(y_bs, pytod_scores[model_name]))
 
                # t-test
                t_stat, p_value = stats.ttest_rel(pyod_bootstrap, pytod_bootstrap)
 
                significance_results.append({
                    'Model': model_name,
                    'PYOD AUC': pyod_auc,
                    'PYTOD AUC': pytod_auc,
                    'Diff': pyod_auc - pytod_auc,
                    'p-value': p_value,
                    'Significant (p<0.05)': 'Yes' if p_value < 0.05 else 'No'
                })
 
                sig_marker = '*' if p_value < 0.05 else ''
                self.logger.info(f"{model_name}: PYOD={pyod_auc:.4f}, PYTOD={pytod_auc:.4f}, diff={pyod_auc-pytod_auc:.4f}, p={p_value:.4f} {sig_marker}")
 
        df_sig = pd.DataFrame(significance_results)
        if not df_sig.empty:
            self.logger.info("=== Significance results table ===")
            self.logger.info(df_sig.to_string(index=False))
 
        # Сравнение скорости
        pyod_avg_time = df_results[df_results['Library'] == 'PYOD']['Train Time (s)'].mean()
        pytod_avg_time = df_results[df_results['Library'] == 'PYTOD']['Train Time (s)'].mean()
        speedup = pyod_avg_time / pytod_avg_time if pytod_avg_time and pytod_avg_time > 0 else float('nan')
 
        self.logger.info(f"Speed comparison: PYOD avg={pyod_avg_time:.4f}s, PYTOD avg={pytod_avg_time:.4f}s, ratio={speedup:.2f}x")
 
        # Выводы
        pyod_best = df_results[df_results['Library'] == 'PYOD'].sort_values('ROC-AUC', ascending=False)
        pytod_best = df_results[df_results['Library'] == 'PYTOD'].sort_values('ROC-AUC', ascending=False)
 
        if not pyod_best.empty:
            best_pyod = pyod_best.iloc[0]
            self.logger.info(f"Best PYOD model: {best_pyod['Model']} (ROC-AUC={best_pyod['ROC-AUC']:.4f})")
        else:
            self.logger.warning('No PYOD models were trained successfully')
 
        if not pytod_best.empty:
            best_pytod = pytod_best.iloc[0]
            self.logger.info(f"Best PYTOD model: {best_pytod['Model']} (ROC-AUC={best_pytod['ROC-AUC']:.4f})")
        else:
            self.logger.warning('No PYTOD models were trained successfully')
 
        return df_results.sort_values('ROC-AUC', ascending=False)
