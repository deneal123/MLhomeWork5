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

        pytod_model_specs = {
            'LOF': ('pytod.models.lof', 'LOF', {'n_neighbors': 20, 'contamination': 0.02}),
            'HBOS': ('pytod.models.hbos', 'HBOS', {'n_bins': 10, 'contamination': 0.02}),
            'KNN': ('pytod.models.knn', 'KNN', {'n_neighbors': 5, 'contamination': 0.02}),
            'PCA': ('pytod.models.pca', 'PCA', {'n_components': 5, 'contamination': 0.02}),
        }

        pytod_models = {}
        for name, (module_name, class_name, kwargs) in pytod_model_specs.items():
            try:
                module = __import__(module_name, fromlist=[class_name])
                cls = getattr(module, class_name)
                pytod_models[name] = lambda cls=cls, kwargs=kwargs: cls(**kwargs)
            except Exception as e:
                self.logger.warning(f"PYTOD model {name} unavailable: {e}")
 
        self.logger.info("Training PYTOD models")
        pytod_scores = {}
        import torch

        X_train_torch = torch.from_numpy(X_train_s).float()
        X_test_torch = torch.from_numpy(X_test_s).float()
        for name, model_fn in tqdm(pytod_models.items(), desc='PYTOD', unit='model'):
             try:
                 model = model_fn()
                 start = time.time()
                 model.fit(X_train_torch)
                 train_time = time.time() - start
                 
                 scores = model.decision_function(X_test_torch)
                 if hasattr(scores, 'detach'):
                     scores = scores.detach().cpu().numpy()

                 if scores is None or (isinstance(scores, np.ndarray) and scores.size == 0):
                     raise ValueError('PYTOD decision_function produced invalid scores')

                 roc = roc_auc_score(y_test, scores)
                 pr = average_precision_score(y_test, scores)

                 pytod_scores[name] = scores

                 results.append({
                     'Library': 'PYTOD',
                     'Model': name,
                     'ROC-AUC': roc,
                     'PR-AUC': pr,
                     'Train Time (s)': train_time,
                 })
                 self.logger.info(f"PYTOD {name}: ROC-AUC={roc:.4f}, Time={train_time:.2f}s")
             except Exception as e:
                 self.logger.error(f"PYTOD {name} failed: {e}")

        df_results = pd.DataFrame(results)
        
        self.logger.info("=== Task4 Results Summary ===")
        self.logger.info(df_results.to_string(index=False))
 
        self.logger.info("=== Statistical Significance (paired t-test) ===")

        def safe_roc(y_true, scores):
            try:
                return roc_auc_score(y_true, scores)
            except Exception:
                return float('nan')

        def safe_pr(y_true, scores):
            try:
                return average_precision_score(y_true, scores)
            except Exception:
                return float('nan')

        significance_results = []
        for model_name in sorted(set(pyod_scores.keys()) & set(pytod_scores.keys())):
             pyod_auc = safe_roc(y_test, pyod_scores[model_name])
             pytod_auc = safe_roc(y_test, pytod_scores[model_name])

             if np.isnan(pyod_auc) or np.isnan(pytod_auc):
                 self.logger.warning(f"Skipping significance for {model_name}: invalid AUC values (pyod={pyod_auc}, pytod={pytod_auc})")
                 continue
             
             n_bootstrap = 100
             pyod_bootstrap = []
             pytod_bootstrap = []

             for _ in range(n_bootstrap):
                 idx = np.random.choice(len(X_test_s), size=len(X_test_s), replace=True)
                 y_bs = y_test.iloc[idx] if hasattr(y_test, 'iloc') else y_test[idx]
                 try:
                     pyod_bootstrap.append(roc_auc_score(y_bs, pyod_scores[model_name][idx]))
                 except Exception:
                     pass
                 try:
                     pytod_bootstrap.append(roc_auc_score(y_bs, pytod_scores[model_name][idx]))
                 except Exception:
                     pass

             if len(pyod_bootstrap) > 1 and len(pytod_bootstrap) > 1 and len(pyod_bootstrap) == len(pytod_bootstrap):
                 _, p_value = stats.ttest_rel(pyod_bootstrap, pytod_bootstrap)
             else:
                 p_value = float('nan')

             significance_results.append({
                 'Model': model_name,
                 'PYOD AUC': pyod_auc,
                 'PYTOD AUC': pytod_auc,
                 'Diff': pyod_auc - pytod_auc,
                 'p-value': p_value,
                 'Significant (p<0.05)': 'Yes' if not np.isnan(p_value) and p_value < 0.05 else 'No'
             })

             sig_marker = '*' if not np.isnan(p_value) and p_value < 0.05 else ''
             self.logger.info(
                 f"{model_name}: PYOD={pyod_auc:.4f}, PYTOD={pytod_auc:.4f}, diff={pyod_auc-pytod_auc:.4f}, p={p_value:.4f} {sig_marker}"
             )

        df_sig = pd.DataFrame(significance_results)
        if df_sig.empty:
            self.logger.info('No paired PYOD-PYTOD comparisons available for significance testing')
        else:
            self.logger.info("=== Significance results table ===")
            self.logger.info(df_sig.to_string(index=False))

        pyod_avg_time = df_results[df_results['Library'] == 'PYOD']['Train Time (s)'].mean() if not df_results[df_results['Library'] == 'PYOD'].empty else float('nan')
        pytod_avg_time = df_results[df_results['Library'] == 'PYTOD']['Train Time (s)'].mean() if not df_results[df_results['Library'] == 'PYTOD'].empty else float('nan')
        if np.isnan(pyod_avg_time):
            pyod_avg_time = 0.0
        if np.isnan(pytod_avg_time):
            pytod_avg_time = 0.0

        if pytod_avg_time > 0:
            speedup = pyod_avg_time / pytod_avg_time
            speed_ratio_str = f"{speedup:.2f}x"
        elif pyod_avg_time > 0 and pytod_avg_time == 0:
            speedup = float('inf')
            speed_ratio_str = "inf"
        else:
            speedup = float('nan')
            speed_ratio_str = "n/a"

        self.logger.info(f"Speed comparison: PYOD avg={pyod_avg_time:.4f}s, PYTOD avg={pytod_avg_time:.4f}s, ratio={speed_ratio_str}")

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
