
# # 0. Про Антифрод

# #### Что такое антифрод?
# **Антифрод (Anti-fraud)** - система предотвращения мошенничества, в данном случае в финансовом мониторинге.
# 
# #### Устройство антифрода (на примере финмониторинга):
# 
# **1. Правила (Rule-based approach)**
# - Известные схемы мошенничества из профильных чатов
# - Примеры схем для обнала:
#   - **Распыление** - дробление крупной суммы на множество мелких транзакций
#   - **Слом назначения платежа** - подмена реального назначения платежа
#   - **Вексельные схемы** - использование векселей для обхода контроля
#   - **Слом НДС** - махинации с НДС
#   - **Транзит** - проведение средств через цепочку счетов
# 
# **2. Экспертные модели**
# - Регрессии на известных фичах (признаках)
# - Примеры фич:
#   - **Доля контрагентов** - соотношение постоянных и новых партнеров
#   - **КНН (Коэффициент налоговой нагрузки)** - отношение налогов к обороту
#   - **Корпоративные карты** - паттерны использования
#   - **Учредитель - подставное лицо** - признаки фиктивности
# 
# **3. Проблема: фичи "ломаются"**
# Нарушители адаптируются:
# - КНН можно искусственно увеличить, отправляя ошибочные платёжки в налоговую и получая возврат
# - Постоянно придумываются новые схемы обхода
# 
# 
# [Актуальный свод статей на тему антифрода](https://github.com/safe-graph/graph-fraud-detection-papers?tab=readme-ov-file#non-gnn-papers-since-2014-back-to-top)

# # 1. Классические алгоритмы. PYOD (Pyton Outlier Detection)
# 
# Библиотека для поиска аномалий на Python
# 
# ### Основные классические алгоритмы
# 
# #### 1. **Isolation Forest (IForest, iForest)**
# - **Идея:** изолировать аномалии через случайные разрезы в пространстве признаков
# - **Механизм:** аномалии легче изолировать (меньше разрезов нужно)
# - **Тип:** бэггинг (ансамбль деревьев)
# - **Плюсы:** быстрый, эффективный, масштабируемый
# - **Когда использовать:** практически всегда хорошая baseline
# 
# #### 2. **LOF (Local Outlier Factor)**
# - **Расшифровка:** Local Outlier Factor - локальный фактор выброса
# - **Идея:** сравнивает локальную плотность точки с плотностью её соседей
# - **Механизм:** если точка в менее плотной области, чем соседи → выброс
# - **Плюсы:** хорош для кластеров разной плотности
# - **Минусы:** медленный на больших данных
# 
# #### 3. **DBSCAN (Density-Based Spatial Clustering)**
# - **Расшифровка:** плотностная кластеризация
# - **Идея:** точки, не попавшие ни в один плотный кластер, считаются выбросами
# - **Параметры:**
#   - `eps` - радиус окрестности
#   - `min_samples` - минимум точек для формирования кластера
# - **Плюсы:** не требует указания числа кластеров
# - **Применение:** когда есть чёткие плотные кластеры
# 
# #### 4. **HBOS (Histogram-Based Outlier Score)**
# - **Расшифровка:** оценка выбросов на основе гистограмм
# - **Идея:** строятся гистограммы для каждого признака, аномалии попадают в редкие бины
# - **Плюсы:** очень быстрый
# - **Минусы:** не учитывает корреляции между признаками
# 
# #### 5. **KNN (K-Nearest Neighbors)**
# - **Идея:** расстояние до k ближайших соседей как мера аномальности
# - **Механизм:** если k соседей далеко → точка аномальна
# - **Плюсы:** простой и понятный
# - **Минусы:** медленный на больших данных
# 
# #### 6. **One-Class SVM**
# - **Идея:** строит границу вокруг "нормальных" данных
# - **Механизм:** всё, что за границей → аномалия
# - **Плюсы:** работает в высоких размерностях
# - **Минусы:** долгое обучение

# In[1]:
# !pip install pyod combo -q

# [Ссылка на туториал из самой библиотеки](https://github.com/yzhao062/pyod/blob/master/notebooks/Compare%20All%20Models.ipynb)

# In[2]:
from __future__ import division
from __future__ import print_function

import os
import sys
from time import time

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

import numpy as np
from numpy import percentile
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn.metrics import recall_score

# Import all models
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.lscp import LSCP
from pyod.models.inne import INNE
from pyod.models.gmm import GMM
from pyod.models.kde import KDE
from pyod.models.lmdd import LMDD
from pyod.models.vae import VAE

import warnings
warnings.filterwarnings("ignore")

# In[3]:
# Define the number of inliers and outliers
n_samples = 200
outliers_fraction = 0.25
clusters_separation = [0]

# Compare given detectors under given settings
# Initialize the data
xx, yy = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-7, 7, 100))
n_inliers = int((1. - outliers_fraction) * n_samples)
n_outliers = int(outliers_fraction * n_samples)
ground_truth = np.zeros(n_samples, dtype=int)
ground_truth[-n_outliers:] = 1

# initialize a set of detectors for LSCP
detector_list = [LOF(n_neighbors=5), LOF(n_neighbors=10), LOF(n_neighbors=15),
                 LOF(n_neighbors=20), LOF(n_neighbors=25), LOF(n_neighbors=30),
                 LOF(n_neighbors=35), LOF(n_neighbors=40), LOF(n_neighbors=45),
                 LOF(n_neighbors=50)]

# In[4]:
random_state = np.random.RandomState(42)
# Define nine outlier detection tools to be compared
classifiers = {
    'Angle-based Outlier Detector (ABOD)':
        ABOD(contamination=outliers_fraction),
    'Cluster-based LOF (CBLOF)':
        CBLOF(contamination=outliers_fraction,
              check_estimator=False, random_state=random_state),
    'Feature Bagging':
        FeatureBagging(LOF(n_neighbors=35),
                       contamination=outliers_fraction,
                       random_state=random_state),
    'Histogram-base OD (HBOS)': HBOS(
        contamination=outliers_fraction),
    'Isolation Forest': IForest(contamination=outliers_fraction,
                                random_state=random_state),
    'KNN': KNN(
        contamination=outliers_fraction),
    'Average KNN': KNN(method='mean',
                       contamination=outliers_fraction),
    'Local OF (LOF)':
        LOF(n_neighbors=35, contamination=outliers_fraction),
    'Min Cov Determinant (MCD)': MCD(
        contamination=outliers_fraction, random_state=random_state),
    'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction),
    'PCA': PCA(
        contamination=outliers_fraction, random_state=random_state),
    'Locally Selective Combination (LSCP)': LSCP(
        detector_list, contamination=outliers_fraction,
        random_state=random_state),
    'INNE': INNE(contamination=outliers_fraction),
    'GMM': GMM(contamination=outliers_fraction),
    'KDE': KDE(contamination=outliers_fraction),
    'LMDD': LMDD(contamination=outliers_fraction)
}

# In[5]:
# Fit the models with the generated data and
# compare model performances
for i, offset in enumerate(clusters_separation):
    np.random.seed(42)
    # Data generation
    X1 = 0.3 * np.random.randn(n_inliers // 2, 2) - offset
    X2 = 0.3 * np.random.randn(n_inliers // 2, 2) + offset
    X = np.r_[X1, X2]
    # Add outliers
    X = np.r_[X, np.random.uniform(low=-6, high=6, size=(n_outliers, 2))]

    # Fit the model
    plt.figure(figsize=(15, 16))
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        # fit the data and tag outliers
        clf.fit(X)
        scores_pred = clf.decision_function(X) * -1
        y_pred = clf.predict(X)
        threshold = percentile(scores_pred, 100 * outliers_fraction)

        print(i + 1, clf_name, recall_score(ground_truth, y_pred))
        n_errors = (y_pred != ground_truth).sum()
        # plot the levels lines and the points

        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
        Z = Z.reshape(xx.shape)
        subplot = plt.subplot(4, 4, i + 1)
        subplot.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),
                         cmap=plt.cm.Blues_r)
        # a = subplot.contour(xx, yy, Z, levels=[threshold],
        #                     linewidths=2, colors='red')
        subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()],
                         colors='orange')
        b = subplot.scatter(X[:-n_outliers, 0], X[:-n_outliers, 1], c='white',
                            s=20, edgecolor='k')
        c = subplot.scatter(X[-n_outliers:, 0], X[-n_outliers:, 1], c='black',
                            s=20, edgecolor='k')
        subplot.axis('tight')
        subplot.legend(
            [
                # a.collections[0],
                b, c],
            [
                # 'learned decision function',
                'true inliers', 'true outliers'],
            prop=matplotlib.font_manager.FontProperties(size=10),
            loc='lower right')
        subplot.set_xlabel("%d. %s (errors: %d)" % (i + 1, clf_name, n_errors))
        subplot.set_xlim((-7, 7))
        subplot.set_ylim((-7, 7))
    plt.subplots_adjust(0.04, 0.1, 0.96, 0.94, 0.1, 0.26)
    plt.suptitle("Outlier detection")
plt.show()

# # PYTOD
# 
# [PYTOD](https://github.com/yzhao062/pytod) - библиотека от того же автора.  
# По сути, это PYOD, но с использованием тензорных операций для ускореня.  
# 
# 
# # Почему это плохо работает на практике?
# 
# **Ответы:**
# 1. **Нет интерпретируемости** - сложно объяснить бизнесу, почему что-то аномалия
# 2. **Плохо работают в высоких размерностях** - curse of dimensionality (проклятие размерности)
# 3. **Требуют хорошего feature engineering** - нужно вручную создавать признаки
# 4. **Не учитывают сложные зависимости** - линейные или простые нелинейные связи, нужно сильно думать над фичами, чтобы эту нелинейность добыть
# 
# 
# **НО:** они всё ещё актуальны как baseline и при хорошем пространстве фичей!  
# 
# Отдельно стоит упомянуть bagging (обучение моделей на подвыборках) и isolation forest.  

# # AutoEncoder (AE)
# 
# **AutoEncoder** - нейросеть, которая сжимает данные в латентное представление и восстанавливает обратно.
# 
# **Архитектура:**
# ```
# Input → Encoder → Latent Space (bottleneck) → Decoder → Output
# ```
# 
# **Идея детекции аномалий:**
# - Обучаем AE на **нормальных** данных
# - Аномалии плохо восстанавливаются
# - **Reconstruction error** (ошибка восстановления) высокая → аномалия
# 
# **Метрика:**
# 
# `score = ||X - X̂||²  (норма разности оригинала и реконструкции)`
# 

# # Датасет транзакций кредитных карт
# 
# Данные:  
# * 29 анонимизированных фичей
# * кванты времени
# * сумма транзакции
# * принадлежность фроду

# In[6]:
# !git clone https://github.com/jeffprosise/Machine-Learning.git

# In[7]:
import pandas as pd

data = pd.read_csv('./Machine-Learning/Data/creditcard.zip')
X, ground_truth = data.drop('Class', axis = 1).copy(), data['Class']
data.head()

# In[8]:
import torch
torch.cuda.is_available()

# In[9]:
from pyod.models.auto_encoder import AutoEncoder

ae = AutoEncoder(device='cuda', epoch_num=10)
ae.fit(X)
y_pred = ae.decision_function(X)

# In[10]:
from sklearn.metrics import roc_auc_score
roc_auc_score(ground_truth, y_pred)

# In[11]:
from sklearn.metrics import PrecisionRecallDisplay

PrecisionRecallDisplay.from_predictions(ground_truth, y_pred, name="AE_default")

# In[12]:
ground_truth.value_counts()

# ## ROC AUC не показателен для несбалансированных классов!
# 
# **ROC AUC (Receiver Operating Characteristic - Area Under Curve)** - площадь под ROC-кривой.
# 
# **Проблема:**
# При сильном дисбалансе классов (например, 1% аномалий) ROC AUC может быть высоким, даже если модель плохо детектирует аномалии.
# 
# **Альтернативы:**
# - **PR AUC** (Precision-Recall AUC) - лучше для несбалансированных данных
# - **F1-score** на заданном threshold
# - **Average Precision**

# # Variational Autoencoder (VAE)
# 
# **VAE** - вариационный автоэнкодер, вероятностная версия AE.
# 
# **Отличия от AE:**
# - Кодирует не в точку, а в распределение (среднее μ и дисперсия σ)
# - Латентное пространство более "гладкое" и организованное
# - Использует KL-divergence в функции потерь
# 
# **Loss:**
# ```
# L = Reconstruction Loss + KL Divergence
# L = ||X - X̂||² + KL(q(z|x) || p(z))
# ```
# 
# **Преимущества для anomaly detection:**
# - Лучше генерализация
# - Более стабильное латентное пространство
# - Может оценивать likelihood (правдоподобие) точки

# # 2. Графовые алгоритмы для детекции аномалий

# # PyGOD - Python Graph Outlier Detection
# 
# **PyGOD** - библиотека для обнаружения аномалий в графах, содержит 10+ алгоритмов.
# 
# GitHub: https://github.com/pygod-team/pygod  
# Docs: https://docs.pygod.org

# In[13]:
# !pip install torch_geometric -q

# аккуратно подберите зависимости

# !pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cpu.html -q

# можно ставить и через !pip install pygod
# но нам нужны бенчмарки, поэтому так

# !git clone https://github.com/pygod-team/pygod.git
# !pip install ./pygod -q

# для табличных данных
# !pip install pyod -q

# In[14]:
import torch

print(torch.cuda.is_available())

torch.manual_seed(45)
torch.cuda.manual_seed(45)
torch.use_deterministic_algorithms(mode=True)

# ## Типы аномалий в графах
# 
# ### 1. **Структурные аномалии (Structural outliers)**
# - Необычная **структура связей**
# - Примеры:
#   - Узел с аномально большим числом рёбер (high degree)
#   - Узел с аномально малым числом рёбер (isolated node)
#   - "Мосты" между сообществами
# 
# ### 2. **Контекстные аномалии (Contextual/Attributed outliers)**
# - Необычные **атрибуты узлов** (features)
# - Примеры:
#   - Пользователь с необычным профилем
#   - Транзакция с аномальной суммой
# 
# ### 3. **Комбинированные аномалии**
# - И структура, и атрибуты аномальны

# ## **GAE (Graph AutoEncoder, 2016)**
# 
# **Архитектура:**
# ```
# A, X → GNN Encoder → Z (node embeddings) → Decoder → Â (reconstructed adjacency)
# ```
# 
# **Компоненты:**
# - **A** - матрица смежности (adjacency matrix)
# - **X** - матрица признаков узлов (node features)
# - **GNN** - Graph Neural Network (GCN, GraphSAGE, GAT и т.д.)
# - **Z** - эмбеддинги узлов (latent representations)
# - **Â** - восстановленная матрица смежности
# 
# **Loss:**
# ```
# L = ||A - Â||**2
# ```
# 
# **Anomaly score:**
# ```
# score(v_i) = ||a_i - â_i||²  (ошибка восстановления связей узла i)
# ```
# 
# **Применение:** детекция структурных аномалий

# ## **VGAE (Variational Graph AutoEncoder, 2016)**
# 
# Вариационная версия GAE.
# 
# **Архитектура:**
# ```
# A, X → GNN Encoder → (μ, σ) → Sample Z ~ N(μ, σ²) → Decoder → Â
# ```
# 
# **Loss:**
# ```
# L = ||A - Â||²_F + KL(q(Z|A,X) || p(Z))
# ```
# 
# **Преимущества:**
# - Более стабильное латентное пространство
# - Может оценивать вероятность
# - Лучше для малых данных

# ## **CONAD (Contrastive Attributed Network Anomaly Detection)**
# 
# **Ключевая идея:** контрастное обучение
# 
# **Механизм:**
# 1. Создаём **позитивные пары** - аугментации одного и того же графа
# 2. Создаём **негативные пары** - разные графы/узлы
# 3. Учим модель различать похожее от непохожего
# 4. Аномалии - то, что не похоже на нормальные паттерны
# 
# **Loss (упрощённо):**
# ```
# L_conad = η · L_cl + (1 − η) * L_recon
# 
# L_cl - contrstive loss:
# - siamesse loss
# - triplet loss
# 
# L_recon - reconstruction loss
# 
# ```
# 
# **Преимущества:**
# - Самообучение (self-supervised)
# - Не требует меток аномалий
# - Учитывает и структуру, и атрибуты
# 
# **Параметры в notebook:** CONAD_02, CONAD_08 - разные веса alpha
# 

# ## **AnomalyDAE (Deep Anomaly Detection on Attributed Networks)**
# 
# **Статья:** [Ding et al., SDM 2019](http://www.public.asu.edu/~kding9/pdf/SDM2019_Deep.pdf)
# 
# [Линк2](https://nsfocusglobal.com/attributed-graph-based-anomaly-detection-and-its-application-in-cybersecurity/)
# 
# **Архитектура:** AutoEncoder с особым дизайном для графов
# 
# **Компоненты:**
# - Encoder для атрибутов узлов
# - Encoder для структуры графа
# - Совместное латентное представление
# - Dual decoder - восстанавливает и X, и A
# 
# **Преимущества:**
# - Обнаруживает оба типа аномалий
# - Может быть настроен на тип аномалий (структурные/контекстные)

# ## DOMINANT model
# 
# **Архитектура:** Dual AutoEncoder
# 
# **Ключевая идея:** раздельное восстановление структуры и атрибутов
# 
# **Loss:**
# ```
# L = (1 - α) * R_s + α * R_A
# R_s = ||A - Â||²_F  (structure reconstruction)
# R_A = ||X - X̂||²_F  (attribute reconstruction)
# ```
# 
# **Параметр α:**
# - α = 0 → только структурные аномалии
# - α = 1 → только контекстные аномалии
# - α = 0.5 → баланс
# 
# **Anomaly score для узла i:**
# ```
# score(v_i) = (1 - α)||a_i - â_i||² + α||x_i - x̂_i||²
# ```
# 
# **Преимущества:**
# - Гибкость - можно настроить на тип аномалий
# - Интерпретируемость - видно, что именно аномально

# ## Бенчмарк PYGOD
# [Бенчмарк](https://github.com/pygod-team/pygod/tree/main/benchmark)

# In[15]:
# ! python pygod/benchmark/main.py --dataset inj_cora --model dominant --gpu 0

# In[16]:
# ! python pygod/benchmark/main.py --dataset inj_cora --model conad --gpu 0

# ## 2.1. PyGOD in notebook

# Datasets description  
# https://github.com/pygod-team/data  

# In[17]:
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import pyod
from pygod.utils import load_data
from sklearn.ensemble import IsolationForest
from pyod.models.lof import LOF
from pygod.detector import DOMINANT, AnomalyDAE, CONAD, GAE

# In[18]:
def calc_node_degree(data):
    s = pd.DataFrame(data.edge_index.numpy().T).groupby(0)[1].count()

    degree = np.zeros(data.x.shape[0],dtype=int)
    degree[s.index] = s.values

    return degree

# In[19]:
def describe_data(data):
    print('========================')

    print(f'Number of nodes: {data.x.shape[0]}')
    print(f'Number of edges: {data.edge_index.shape[1]}')
    print(f'Number of features: {data.x.shape[1]}')

    if data.edge_attr is not None:
        print(f'Number of edge attributes: {data.edge_attr.shape[1]}')

    y = data.y.numpy()
    y = y[~np.isnan(y) & (y!=-100)]
    print(f'Number of classes: {len(np.unique(y))}')
    print('========================')

# In[20]:
def print_gini(data, score, model_type = ''):
    if not isinstance(score,np.ndarray):
        score = score.numpy()

    structural_mask = np.isin(data.y.numpy(),([0,2,3]))
    contextual_mask = np.isin(data.y.numpy(),([0,1,3]))
    y = data.y.bool().numpy().astype(int)

    gini_structural =  2*roc_auc_score(y[structural_mask], score[structural_mask])-1
    gini_contextual =  2*roc_auc_score(y[contextual_mask], score[contextual_mask])-1
    gini_total = 2*roc_auc_score(y, score)-1

    print("""GINI structural = {:.3f}
GINI contextual = {:.3f}
GINI total      = {:.3f}
    """.format(gini_structural, gini_contextual, gini_total))
    a, b, c = [round(x, 3) for x in [gini_structural, gini_contextual, gini_total]]
    return model_type, a, b, c

# In[21]:
def test_all_models(datasets):
  res_all = pd.DataFrame()
  for dataset in datasets:
    res = []
    data = load_data(dataset)
    describe_data(data)
    # - 0: inlier
    # - 1: contextual outlier only
    # - 2: structural outlier only
    # - 3: both contextual outlier and structural outlier

    # labels count
    # число представителей класса
    print(pd.Series(data.y.numpy()).value_counts())

    print('degree')
    degree = calc_node_degree(data)
    res.append(print_gini(data, degree, 'node_degree'))

    # without graph
    model = IsolationForest()
    model.fit(data.x.numpy())
    score = model.decision_function(data.x.numpy())
    print('IsolationForest')
    res.append(print_gini(data, score, 'IsolationForest'))

    # without graph
    model = LOF()
    model.fit(data.x.numpy())
    score = model.decision_function(data.x.numpy())
    print('LOF')
    res.append(print_gini(data, score, 'LOF'))

    # with graph
    gpu = 0 # <=> cuda
    # gpu = -1 # <=> cpu

    model = DOMINANT(gpu=gpu)
    model.fit(data)
    score = model.decision_score_.numpy()
    print('DOMINANT')
    res.append(print_gini(data, score, 'DOMINANT'))

    model = CONAD(gpu=gpu)
    model.fit(data)
    score = model.decision_score_.numpy()
    y =data.y.bool().numpy().astype(int)
    print('CONAD')
    res.append(print_gini(data, score, 'CONAD'))

    model = CONAD(gpu=gpu, eta = 0.2)
    model.fit(data)
    score = model.decision_score_.numpy()
    y =data.y.bool().numpy().astype(int)
    print('CONAD')
    res.append(print_gini(data, score, 'CONAD_02'))

    model = CONAD(gpu=gpu, eta = 0.8)
    model.fit(data)
    score = model.decision_score_.numpy()
    y =data.y.bool().numpy().astype(int)
    print('CONAD')
    res.append(print_gini(data, score, 'CONAD_08'))

    model = AnomalyDAE()
    model.fit(data)
    score = model.decision_score_.numpy()
    y =data.y.bool().numpy().astype(int)
    print('AnomalyDAE')
    res.append(print_gini(data, score, 'AnomalyDAE'))

    model = GAE(gpu=gpu)
    model.fit(data)
    score = model.decision_score_.numpy()
    y =data.y.bool().numpy().astype(int)
    print('GAE')
    res.append(print_gini(data, score, 'GAE'))

    res_pd = pd.DataFrame(res, columns = ['model', 'gini structural', 'gini contextual', 'gini total '])
    res_pd['dataset'] = dataset

    if res_all.shape[1] == 1:
      res_all = res_pd.copy()
    else:
      res_all = pd.concat([res_all, res_pd], ignore_index = True)
  return res_all

# In[22]:
test_all_models(['inj_cora', 'inj_amazon'])

# ## Датасет weibo

# Weibo has organic outliers
# 
# Dataset from
# Error-bounded graph anomaly loss for gnns.
# T. Zhao, C. Deng, K. Yu, T. Jiang, D. Wang, and M. Jiang.
# In Proceedings of the 29th ACM International Conference on Information & Knowledge Management, pages 1873–1882, 2020.
# https://tzhao.io/files/papers/CIKM20_GAL.pdf
# 
# Tencent-Weibo is a user-posts-hashtag graph from a Twitter-like platform.
# It has 8,405 users
# There is an edge between 2 users if they used same hashtag in post.  
# 100 location features.   
# 300 bag-of-words features.  

# In[23]:
data = load_data('weibo')
describe_data(data)
# Доли классов
pd.Series(data.y.numpy()).value_counts(normalize=True).round(3)

# In[24]:
def print_gini_simple(data, score, model_type = ''):
  if not isinstance(score,np.ndarray):
          score = score.numpy()
  y = data.y.bool().numpy().astype(int)
  gini_total = 2 * roc_auc_score(y, score) - 1
  print(model_type)
  print( f'{gini_total:.3f}')
  return model_type, round(gini_total, 3)

# In[25]:
res = []
degree = calc_node_degree(data)
res.append(print_gini_simple(data, degree, 'node degree'))

model = LOF()
model.fit(data.x.numpy())
score = model.decision_function(data.x.numpy())
res.append(print_gini_simple(data, score, 'LOF'))

gpu = 0
model = AnomalyDAE(gpu=gpu)
model.fit(data)
score = model.decision_score_.numpy()
res.append(print_gini_simple(data, score, 'AnomalyDAE'))

model = CONAD(gpu=gpu)
model.fit(data)
score = model.decision_score_.numpy()
res.append(print_gini_simple(data, score, 'CONAD'))

model = DOMINANT(gpu=gpu)
model.fit(data)
score = model.decision_score_.numpy()
res.append(print_gini_simple(data, score, 'DOMINANT'))

from pygod.detector import Radar
model = Radar(gpu=gpu)

model.fit(data)
score = model.decision_score_.numpy()
res.append(print_gini_simple(data, score, 'Radar'))

# In[26]:
res_pd = pd.DataFrame(res, columns = ['model', 'gini total '])
res_pd

# # 3. Транзакционные эмбеддинги.

# ### COLES (Contrastive Learning for Event Sequences)
# 
# **Статья:** https://arxiv.org/pdf/2002.08232.pdf
# 
# **Полное название:** CoLES - Contrastive Learning for Event Sequences
# 
# **Ключевая идея:** self-supervised обучение на последовательностях событий
# 
# **Механизм:**
# 
# 1. **Augmentation (аугментация) последовательностей:**
#    - Случайные crop (обрезки)
#    - Subsampling (подвыборка событий)
#    - Временные сдвиги
#    
# 2. **Contrastive loss:**
#    - Создаём две аугментации одной последовательности → **позитивная пара**
#    - Берём последовательности разных клиентов → **негативные пары**
#    - Учим энкодер различать их
# 
# 3. **Эмбеддинги клиентов:**
#    - Энкодер превращает последовательность транзакций в вектор
#    - Вектор содержит "поведенческий отпечаток" клиента
# 
# **Применение для anomaly detection:**
# - Обучаем COLES на нормальных последовательностях
# - Аномальные последовательности дают "странные" эмбеддинги
# - Используем классификатор или outlier detection на эмбеддингах

# ### PTLS (PyTorch Lifestream)
# 
# **GitHub:** https://github.com/dllllb/pytorch-lifestream
# 
# Библиотека для работы с последовательностями событий (event sequences)
# 
# **Возможности:**
# - Различные архитектуры энкодеров:
#   - **RNN** (LSTM, GRU)
#   - **Transformer**
#   - **COLES**
# - Аугментации для последовательностей
# - Инструменты для обучения и inference
# - Интеграция с PyTorch Lightning
# 
# **Применение:**
# - Кредитный скоринг
# - Детекция фрода
# - Churn prediction (предсказание оттока)
# - Рекомендательные системы
# 
# **Пайплайны обучения отлично заточены под pytorch_lightning**

# Смените среду выполнения на GPU

# In[27]:
# !pip install -q pytorch-lifestream lightgbm pytorch_lightning==1.9.0

# In[28]:
# !git clone https://github.com/atavci/fraud-detection-on-banksim-data.git

# In[29]:
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List
from functools import partial
from pathlib import Path

from transformers import AutoTokenizer, AutoModel

from ptls.data_load.utils import collate_feature_dict
from ptls.frames.inference_module import InferenceModule
from ptls.nn import TrxEncoder, RnnSeqEncoder
from ptls.preprocessing import PandasDataPreprocessor
from ptls.frames.coles import ColesDataset, CoLESModule
from ptls.frames.coles.split_strategy import SampleSlices
from ptls.frames import PtlsDataModule
from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.iterable_processing import ISeqLenLimit, FeatureFilter
# from ptls.nn.trx_encoder.encoders import IdentityEncoder

from sklearn.preprocessing import MaxAbsScaler

from pytorch_lightning.loggers import TensorBoardLogger

# In[30]:
data = pd.read_csv('fraud-detection-on-banksim-data/Data/synthetic-data-from-a-financial-payment-system/bs140513_032310.csv')

# In[31]:
data.head(5)

# In[32]:
data.info()

# In[33]:
df_fraud = data.loc[data.fraud == 1]
df_non_fraud = data.loc[data.fraud == 0]

sns.countplot(x="fraud",data=data)
plt.title("Count of Fraudulent Payments")
plt.show()
print("Number of normal examples: ",df_non_fraud.fraud.count())
print("Number of fradulent examples: ",df_fraud.fraud.count())

# In[34]:
# Видим, что зип мерчант и зипкоды всегда одинаковые, их можно выкинуть
len(data.merchant.value_counts()), len(data.zipMerchant.value_counts()), len(data.zipcodeOri.value_counts())

data = data.drop(['zipMerchant', 'zipcodeOri'], axis=1)

# In[35]:
data.isna().sum()

# In[36]:
# на каждом степе по 40 фродовых транзакций
data.groupby('step')['fraud'].sum().unique()

# In[37]:
# Число транзакций растет с увеличением степа
data.groupby('step')['fraud'].count().reset_index().plot.scatter(x='step', y='fraud')

# In[38]:
grouped_data = data.groupby('customer').aggregate({
    'step': 'count',
    'fraud': 'sum'
    }).rename({'step': 'num'}, axis=1)
grouped_data['has_fraud'] = grouped_data['fraud'] > 0

plt.title('num transactions')
plt.hist(grouped_data[grouped_data.has_fraud].num, bins=40, label='fraud_customers', alpha=0.5)
plt.hist(grouped_data[~grouped_data.has_fraud].num, bins=40, label='good_customers', alpha=0.5)
plt.legend();

# In[39]:
# Посмотрим на число фродовых транзакций у тех, у кого есть хотя бы одна
plt.title('Num fraud transactions for customers with at least one fraud')
plt.hist(grouped_data[grouped_data.has_fraud].fraud.clip(0, 30), bins=31);

# Сделаем train/val сплит по клиентам а не мерчантам

# In[40]:
(grouped_data[grouped_data.has_fraud]['fraud'] > 1).mean()

# In[41]:
from sklearn.model_selection import train_test_split
train_customers, val_customers = train_test_split(
    grouped_data.index, test_size=0.1, random_state=42, stratify=grouped_data.has_fraud
)

train_df= data[data.customer.isin(train_customers)]
val_df = data[data.customer.isin(val_customers)]

# In[42]:
train_df.shape, val_df.shape

# In[43]:
train_df['fraud'].sum(), val_df['fraud'].sum()

# In[44]:
X_train, y_train = train_df.drop('fraud', axis=1), train_df['fraud']
X_val, y_val = val_df.drop('fraud', axis=1), val_df['fraud']

# In[45]:
# from lightgbm import LGBMClassifier

preprocessor = PandasDataPreprocessor(
    col_id='customer',
    col_event_time='step',
    event_time_transformation='none',
    cols_category=['gender','age','merchant','category'],
    cols_numerical=['amount'],
    cols_identity=[],
)
X_train_coles = MemoryMapDataset(
    data=preprocessor.fit_transform(X_train),
    i_filters=[
        ISeqLenLimit(max_seq_len=200),
    ]
)
X_val_coles = MemoryMapDataset(
    data=preprocessor.transform(X_val),
    i_filters=[
        ISeqLenLimit(max_seq_len=200),
    ]
)

# In[46]:
trx_encoder_params = dict(
    embeddings_noise=0.003,
    numeric_values={'amount': 'identity'},
    embeddings={
        'step': {'in': 800, 'out': 16},
    },
    norm_embeddings=False
)

seq_encoder = RnnSeqEncoder(
    trx_encoder=TrxEncoder(**trx_encoder_params),
    hidden_size=256,
    type='gru',
    bidir=False,
    trainable_starter='static'
)

model = CoLESModule(
    seq_encoder=seq_encoder,
    optimizer_partial=partial(torch.optim.Adam, lr=0.001, weight_decay=0.0),
    lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=30, gamma=0.9),
)
train_dl = PtlsDataModule(
    train_data=ColesDataset(
        X_train_coles,
        splitter=SampleSlices(
            split_count=5,
            cnt_min=15,
            cnt_max=75,
        ),
    ),
    train_num_workers=4, #2
    train_batch_size=256, #64
    valid_data=ColesDataset(
        X_val_coles,
        splitter=SampleSlices(
            split_count=5,
            cnt_min=25,
            cnt_max=200
        )
    ),
    valid_batch_size=256,
    valid_num_workers=4
)

# In[47]:
trainer = pl.Trainer(
    max_epochs=200,
    gpus=1,
    enable_progress_bar=True,
)

# In[48]:
# %%time
print(trainer.logger.version)
trainer.fit(model, train_dl)
print(trainer.logged_metrics)

# In[49]:
train_dl = torch.utils.data.DataLoader(
    dataset=X_train_coles,
    collate_fn=collate_feature_dict,
    shuffle=False,
    batch_size=512,
    num_workers=4 #2
)

val_dl = torch.utils.data.DataLoader(
    dataset=X_val_coles,
    collate_fn=collate_feature_dict,
    shuffle=False,
    batch_size=512,
    num_workers=4 #2
)
inf_model = InferenceModule(
    seq_encoder
)

# In[50]:
coles_train = pd.concat(trainer.predict(inf_model, train_dl))
coles_val = pd.concat(trainer.predict(inf_model, val_dl))

# In[51]:
def encode_categorical(df_train, col_name, df_val=None):
    # Для каждого значения категориальной переменной мы посчитаем mean, std amount для строк
    # с таким значением перемнной и заменим колонку с категор фичей на эти две колонки
    stats = data.groupby(col_name).agg({'amount': ['mean', 'std']}).droplevel(0, axis=1)
    stats = stats.rename({
        'mean': col_name + '_stats_mean',
        'std': col_name + '_stats_std'
    },axis=1)

    if df_val is None:
        return df_train.merge(stats, left_on=col_name, right_index=True).drop(col_name,axis=1)
    else:
        return (df_train.merge(stats, left_on=col_name, right_index=True).drop(col_name,axis=1),
                df_val.merge(stats, left_on=col_name, right_index=True).drop(col_name,axis=1))

# In[52]:
X_train_enc, X_val_enc = encode_categorical(X_train, 'gender', X_val)
X_train_enc, X_val_enc = encode_categorical(X_train_enc, 'category', X_val_enc)
X_train_enc, X_val_enc = encode_categorical(X_train_enc, 'age', X_val_enc)
X_train_enc, X_val_enc = encode_categorical(X_train_enc, 'merchant', X_val_enc)
X_train_enc = X_train_enc.merge(coles_train, on='customer')
X_val_enc = X_val_enc.merge(coles_val, on='customer')
X_train_enc = X_train_enc.drop(['step', 'customer'], axis=1)
X_val_enc = X_val_enc.drop(['step', 'customer'], axis=1)

# In[53]:
import pickle
with open('X_train_enc.pickle', 'wb') as handle:
  pickle.dump(X_train_enc, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('X_val_enc.pickle', 'wb') as handle:
  pickle.dump(X_val_enc, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('y_val.pickle', 'wb') as handle:
  pickle.dump(y_val, handle, protocol=pickle.HIGHEST_PROTOCOL)

# In[54]:
# !pip install pyod

# In[55]:
with open('X_train_enc.pickle', 'rb') as handle:
  X_train_enc = pickle.load(handle)
with open('X_val_enc.pickle', 'rb') as handle:
  X_val_enc = pickle.load(handle)
with open('y_val.pickle', 'rb') as handle:
  y_val = pickle.load(handle)

# In[56]:
# from pyod.models.auto_encoder_torch import AutoEncoder
from pyod.models.vae import VAE

ae = VAE(epoch_num = 3,  latent_dim=15)
ae.fit(X_train_enc)

# In[57]:
y_val_pred = ae.decision_function(X_val_enc)
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score
PrecisionRecallDisplay.from_predictions(y_val, y_val_pred, name="COLES+AE")

# In[58]:
('COLES+AE', average_precision_score(y_val, y_val_pred))

# # 3. Домашнее задание

# **На выбор:**
# - Сравнение VAE vs COLES vs Classic OD (IFOrest, DBSCAN, etc.) на датасете №1
# - Сравнить качество работы unsupervised детекторов аномалий на датасете №2
# - Разбор на предмет аномалий датасета №3   
# - Сравнить PYOD vs PYTOD (10 алгоритмов) на датасете №1
# 
# **Датасеты:**
# 1. Фродовые транзакции - https://github.com/atavci/fraud-detection-on-banksim-data/blob/master/Data/  
# 2. Фродовые транзакции -  https://github.com/jeffprosise/Machine-Learning/blob/master/Data/creditcard.zip  
#     Меньше сэмплов, больше признаков, нет идентификации по клиенту
# 3. Датасет криптовалютных транзакций - https://www.kaggle.com/datasets/ellipticco/elliptic-data-set   
#     [бонус](https://www.kaggle.com/datasets/alexbenzik/deanonymized-995-pct-of-elliptic-transactions)

# ## Полезные ссылки
# 
# ### Библиотеки
# - PyOD: https://github.com/yzhao062/pyod
# - PyTOD: https://github.com/yzhao062/pytod
# - PyGOD: https://github.com/pygod-team/pygod
# - PTLS: https://github.com/dllllb/pytorch-lifestream
# 
# ### Статьи и ресурсы
# - Graph Fraud Detection Papers: https://github.com/safe-graph/graph-fraud-detection-papers
# - COLES paper: https://arxiv.org/pdf/2002.08232.pdf
# - BOND paper: https://www.cs.cmu.edu/~zhihaoj2/papers/Bond_NeurIPS22.pdf
# - AnomalyDAE paper: https://www.public.asu.edu/~kding9/pdf/SDM2019_Deep.pdf
# 
# ### Датасеты
# - PyGOD datasets: https://github.com/pygod-team/data
# - BankSim fraud dataset: https://github.com/atavci/fraud-detection-on-banksim-data

# 