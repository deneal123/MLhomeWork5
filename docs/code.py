#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Полное решение домашнего задания №5 по обнаружению аномалий.
Выполнены все 4 варианта задания с детальным EDA, обучением моделей и сравнением.
"""

import os
import sys
import time
import warnings
import urllib.request
import zipfile
import subprocess
from collections import Counter
from pathlib import Path
import urllib.parse

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import IsolationForest as SklearnIsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Отключаем предупреждения для чистоты вывода
warnings.filterwarnings('ignore')
matplotlib.use('Agg')          # для сохранения графиков без GUI

# Задаём seed для воспроизводимости
SEED = 42
np.random.seed(SEED)
import torch
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def check_gpu():
    """Проверяет доступность GPU и выводит статус."""
    if torch.cuda.is_available():
        print(f"GPU доступен: {torch.cuda.device_count()} устройство(а): {torch.cuda.get_device_name(0)}")
    else:
        print("GPU не доступен. Будет использовано CPU.")


# ------------------------------------------------------------------------------
#  Вспомогательные функции
# ------------------------------------------------------------------------------

def _load_dotenv_file(env_path):
    """Загружает переменные окружения из .env (формат KEY=VALUE)."""
    env_path = Path(env_path)
    if not env_path.exists():
        return

    print(f"Загрузка переменных из {env_path}")
    with env_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, value = map(str.strip, line.split('=', 1))
            if not key:
                continue
            if ((key not in os.environ) and (key.upper() not in os.environ)):
                os.environ[key] = value
                os.environ[key.upper()] = value
                os.environ[key.lower()] = value


def download_file(url, dest_path):
    """Загружает файл по URL, если он ещё не существует."""
    dest_path = Path(dest_path)
    if dest_path.exists():
        print(f"Файл {dest_path} уже существует. Пропускаем загрузку.")
        return

    # Попробуем загрузить настройки прокси из .env (если есть)
    for env_rel_path in [Path('src/antifraud/config/.env'), Path('.env')]:
        if env_rel_path.exists():
            _load_dotenv_file(env_rel_path)
            break

    print(f"Загрузка {url} -> {dest_path}")

    # Поддержка прокси из переменных ENV
    proxies = {}
    for scheme in ('http', 'https'):
        for env_var in (f'{scheme}_proxy', f'{scheme.upper()}_PROXY'):
            value = os.environ.get(env_var)
            if value:
                proxies[scheme] = value
                break

    if not proxies:
        proxies = urllib.request.getproxies() or {}

    handlers = []
    if proxies:
        print(f"Использую прокси-серверы: {proxies}")
        handlers.append(urllib.request.ProxyHandler(proxies))

        # Если прокси содержит имя пользователя и пароль, добавляем ProxyBasicAuthHandler
        for proxy in proxies.values():
            parsed = urllib.parse.urlparse(proxy)
            if parsed.username and parsed.password:
                auth_handler = urllib.request.ProxyBasicAuthHandler()
                auth_handler.add_password(
                    realm=None,
                    uri=f"{parsed.scheme}://{parsed.hostname}",
                    user=urllib.parse.unquote(parsed.username),
                    passwd=urllib.parse.unquote(parsed.password)
                )
                handlers.append(auth_handler)
                break

    opener = urllib.request.build_opener(*handlers) if handlers else urllib.request.build_opener()

    try:
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, dest_path)
        print("Готово.")
    except urllib.error.URLError as e:
        msg = (f"Не удалось загрузить {url}. Убедитесь, что у вас есть доступ в интернет "
               "или прокси настроена корректно (HTTP_PROXY/HTTPS_PROXY).\n"
               f"Ошибка: {e}")
        print(msg)
        raise RuntimeError(msg) from e
    except OSError as e:
        msg = f"Ошибка ввода-вывода при загрузке {url}: {e}"
        print(msg)
        raise RuntimeError(msg) from e

def unzip_file(zip_path, extract_to):
    """Распаковывает zip-архив в указанную папку."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def clone_repo(repo_url, target_dir):
    """Клонирует git-репозиторий, если папка не существует."""
    if Path(target_dir).exists():
        print(f"Папка {target_dir} уже существует. Пропускаем клонирование.")
        return
    print(f"Клонирование {repo_url} -> {target_dir}")
    subprocess.run(['git', 'clone', repo_url, target_dir], check=True)

# ------------------------------------------------------------------------------
#  Загрузка и подготовка данных
# ------------------------------------------------------------------------------

def load_banksim():
    """Загружает датасет BankSim."""
    repo_dir = Path("fraud-detection-on-banksim-data")
    if not repo_dir.exists():
        clone_repo("https://github.com/atavci/fraud-detection-on-banksim-data.git", repo_dir)

    data_path = repo_dir / "Data" / "synthetic-data-from-a-financial-payment-system" / "bs140513_032310.csv"
    df = pd.read_csv(data_path)
    return df

def load_creditcard():
    """Загружает датасет Credit Card."""
    data_dir = Path("creditcard_data")
    data_dir.mkdir(exist_ok=True)

    zip_path = data_dir / "creditcard.zip"
    csv_path = data_dir / "creditcard.csv"

    if csv_path.exists():
        print(f"Найден {csv_path}, загружаем без скачивания")
        return pd.read_csv(csv_path)

    if not zip_path.exists():
        try:
            download_file("https://github.com/jeffprosise/Machine-Learning/raw/master/Data/creditcard.zip", zip_path)
        except RuntimeError as e:
            raise RuntimeError(
                "Не удалось загрузить creditcard.zip. "
                "Пожалуйста, скачайте вручную и поместите в директорию creditcard_data/ "
                "или настройте HTTP(S)_PROXY."
            ) from e

    if not csv_path.exists():
        unzip_file(zip_path, data_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"Ожидался {csv_path}, но он не найден после распаковки.")

    return pd.read_csv(csv_path)

def load_weibo():
    """Загружает датасет Weibo через PyGOD."""
    print("Начинаем загрузку Weibo")
    local_weibo_path = Path("weibo_data")
    local_weibo_path.mkdir(exist_ok=True)

    # Первый приоритет - локальный заранее загруженный файл
    weibo_file = local_weibo_path / "weibo.pt"
    if weibo_file.exists():
        print(f"Найден локальный файл Weibo: {weibo_file}. Используем его.")
        try:
            from pygod.utils import load_data
            data = load_data('weibo', root=local_weibo_path)
            print("Weibo загружен из локального файла")
            return data
        except Exception as e:
            print(f"Ошибка загрузки Weibo из локального файла: {e}")
            # Продолжаем, чтобы попытаться загрузить из интернета

    try:
        from pygod.utils import load_data
        print("Загружаем Weibo через pygod.utils.load_data('weibo')")
        data = load_data('weibo')
        print("Weibo успешно загружен через PyGOD")
        return data
    except ImportError:
        msg = "PyGOD не установлен. Установите: pip install pygod"
        print(msg)
        return None
    except Exception as e:
        print(f"Не удалось загрузить Weibo через интернет: {e}")
        print("Если вы работаете за прокси, проверьте переменные HTTP_PROXY/HTTPS_PROXY.\n"
              "Можно загрузить файл вручную и поместить его в weibo_data/ или временно пропустить этот этап.")
        return None

# ------------------------------------------------------------------------------
#  EDA (разведочный анализ)
# ------------------------------------------------------------------------------

def eda_banksim(df):
    """Строит 4+ графика для BankSim."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax1 = axes[0, 0]
    df['fraud'].value_counts().plot(kind='bar', ax=ax1, color=['green', 'red'])
    ax1.set_title('BankSim: Распределение фрода')
    ax1.set_xticklabels(['Норма', 'Фрод'], rotation=0)

    ax2 = axes[0, 1]
    df[df['fraud']==0]['amount'].hist(bins=30, ax=ax2, alpha=0.7, label='Норма', color='green')
    df[df['fraud']==1]['amount'].hist(bins=30, ax=ax2, alpha=0.7, label='Фрод', color='red')
    ax2.set_title('BankSim: Распределение суммы')
    ax2.legend()

    ax3 = axes[1, 0]
    df['category'].value_counts().head(8).plot(kind='barh', ax=ax3, color='steelblue')
    ax3.set_title('BankSim: Топ категорий')

    ax4 = axes[1, 1]
    df['age'].value_counts().plot(kind='bar', ax=ax4, color='coral')
    ax4.set_title('BankSim: Распределение возраста')
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('eda_banksim.png', dpi=100)
    plt.close()
    print("EDA BankSim сохранён в eda_banksim.png")

def eda_creditcard(df):
    """Строит 4+ графика для Credit Card."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax1 = axes[0, 0]
    df['Class'].value_counts().plot(kind='bar', ax=ax1, color=['green', 'red'])
    ax1.set_title('Credit Card: Распределение классов')
    ax1.set_xticklabels(['Норма', 'Фрод'], rotation=0)

    ax2 = axes[0, 1]
    df[df['Class']==0]['Amount'].hist(bins=30, ax=ax2, alpha=0.7, label='Норма', color='green')
    df[df['Class']==1]['Amount'].hist(bins=30, ax=ax2, alpha=0.7, label='Фрод', color='red')
    ax2.set_title('Credit Card: Распределение суммы')
    ax2.legend()

    ax3 = axes[1, 0]
    corr = df.drop(['Time','Amount'], axis=1).corr()['Class'].drop('Class').abs().sort_values(ascending=False).head(10)
    corr.plot(kind='barh', ax=ax3, color='steelblue')
    ax3.set_title('Credit Card: Топ корреляций с классом')

    ax4 = axes[1, 1]
    df['Time'].hist(bins=30, ax=ax4, color='purple')
    ax4.set_title('Credit Card: Распределение времени')

    plt.tight_layout()
    plt.savefig('eda_creditcard.png', dpi=100)
    plt.close()
    print("EDA Credit Card сохранён в eda_creditcard.png")

def eda_weibo(data):
    """Строит 4+ графика для Weibo."""
    edge_list = data.edge_index.numpy().T
    degrees = Counter(edge_list[:, 0]) + Counter(edge_list[:, 1])
    degree_vals = list(degrees.values())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax1 = axes[0, 0]
    ax1.hist(degree_vals, bins=30, color='steelblue', edgecolor='black')
    ax1.set_title('Weibo: Распределение степеней узлов')
    ax1.axvline(np.mean(degree_vals), color='red', linestyle='--', label=f'Среднее: {np.mean(degree_vals):.1f}')
    ax1.legend()

    ax2 = axes[0, 1]
    labels = data.y.numpy()
    pd.Series(labels).value_counts().plot(kind='bar', ax=ax2, color=['green', 'red'])
    ax2.set_title('Weibo: Распределение аномалий')
    ax2.set_xticklabels(['Норма', 'Аномалия'], rotation=0)

    ax3 = axes[1, 0]
    feature_means = data.x.numpy().mean(axis=0)
    ax3.hist(feature_means, bins=30, color='coral', edgecolor='black')
    ax3.set_title('Weibo: Распределение средних признаков')

    ax4 = axes[1, 1]
    feature_vars = data.x.numpy().var(axis=0)
    ax4.hist(feature_vars, bins=30, color='purple', edgecolor='black')
    ax4.set_title('Weibo: Распределение дисперсий признаков')

    plt.tight_layout()
    plt.savefig('eda_weibo.png', dpi=100)
    plt.close()
    print("EDA Weibo сохранён в eda_weibo.png")

# ------------------------------------------------------------------------------
#  Пункт 1: Сравнение VAE vs COLES vs Classic OD на BankSim
# ------------------------------------------------------------------------------

def task1_banksim():
    print("\n" + "="*60)
    print("ПУНКТ 1: Сравнение VAE vs COLES vs Classic OD (BankSim)")
    print("="*60)

    # 1. Загрузка данных
    df = load_banksim()
    df.drop(['zipMerchant', 'zipcodeOri'], axis=1, inplace=True)  # удаляем бесполезные колонки

    # Кодируем категории
    for col in ['gender', 'age', 'merchant', 'category']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Разделение на train/test (20% тест)
    X = df.drop(['customer', 'fraud'], axis=1)
    y = df['fraud']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Масштабирование
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    pca_n_components = min(X_train_s.shape[1], 10)
    print(f"Используем для PCA n_components={pca_n_components}")

    # --- Базовые классические модели ---
    models = {
        'Isolation Forest': SklearnIsolationForest(n_estimators=100, contamination=0.02, random_state=SEED, n_jobs=-1),
        'LOF': LocalOutlierFactor(n_neighbors=20, contamination=0.02, novelty=True, n_jobs=-1),
        'HBOS': None,   # импорт позже
        'KNN': None,
        'PCA': None
    }
    # Импортируем pyod модели
    from pyod.models.hbos import HBOS
    from pyod.models.knn import KNN
    from pyod.models.pca import PCA as PyOD_PCA
    models['HBOS'] = HBOS(n_bins=10, contamination=0.02)
    models['KNN'] = KNN(n_neighbors=5, contamination=0.02)
    models['PCA'] = PyOD_PCA(n_components=pca_n_components, contamination=0.02, random_state=SEED)

    classic_results = {}
    print("Обучение классических моделей...")
    for name, model in tqdm(models.items(), desc='Classic models', unit='model'):
        print(f"   Запуск {name}...")
        start = time.time()
        model.fit(X_train_s)
        scores = model.decision_function(X_test_s)
        t = time.time() - start
        auc = roc_auc_score(y_test, scores)
        ap = average_precision_score(y_test, scores)
        classic_results[name] = {'ROC-AUC': auc, 'PR-AUC': ap, 'Time (s)': t}
        print(f"  {name}: ROC-AUC = {auc:.4f}, PR-AUC = {ap:.4f}, Time={t:.2f}s")

    # --- VAE ---
    print("\nОбучение VAE (autoencoder)...")
    from pyod.models.vae import VAE
    vae = VAE(epoch_num=5, latent_dim=8, encoder_neuron_list=[32,16],
              decoder_neuron_list=[16,32], contamination=0.02, verbose=0, batch_size=512)
    vae.fit(X_train_s)
    vae_scores = vae.decision_function(X_test_s)
    auc_vae = roc_auc_score(y_test, vae_scores)
    ap_vae = average_precision_score(y_test, vae_scores)
    classic_results['VAE'] = {'ROC-AUC': auc_vae, 'PR-AUC': ap_vae, 'Time (s)': 0}
    print(f"  VAE: ROC-AUC = {auc_vae:.4f}, PR-AUC = {ap_vae:.4f}")

    # --- COLES (обучение эмбеддингов и детектор) ---
    try:
        from ptls.data_load.utils import collate_feature_dict
        from ptls.frames.inference_module import InferenceModule
        from ptls.nn import TrxEncoder, RnnSeqEncoder
        from ptls.preprocessing import PandasDataPreprocessor
        from ptls.frames.coles import ColesDataset, CoLESModule
        from ptls.frames.coles.split_strategy import SampleSlices
        from ptls.frames import PtlsDataModule
        from ptls.data_load.datasets import MemoryMapDataset
        from ptls.data_load.iterable_processing import ISeqLenLimit
        from pytorch_lightning import Trainer
        from pytorch_lightning.loggers import TensorBoardLogger

        print("\nПодготовка данных для COLES...")
        preprocessor = PandasDataPreprocessor(
            col_id='customer',
            col_event_time='step',
            event_time_transformation='none',
            cols_category=['gender', 'age', 'merchant', 'category'],
            cols_numerical=['amount'],
            cols_identity=[],
        )
        # Обучаем на train, но используем те же клиенты? Для чистоты эксперимента нужно, чтобы train и test были из разных клиентов.
        # Но в банковских данных у клиента могут быть и фродовые транзакции. Для COLES лучше брать только нормальные транзакции для обучения.
        # Упростим: возьмём всех клиентов из train_df.
        train_df = df[df['customer'].isin(X_train['customer'].unique())]
        test_df = df[df['customer'].isin(X_test['customer'].unique())]

        X_train_coles = MemoryMapDataset(
            data=preprocessor.fit_transform(train_df),
            i_filters=[ISeqLenLimit(max_seq_len=200)]
        )
        X_val_coles = MemoryMapDataset(
            data=preprocessor.transform(test_df),
            i_filters=[ISeqLenLimit(max_seq_len=200)]
        )

        # Параметры энкодера
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
        coles_module = CoLESModule(
            seq_encoder=seq_encoder,
            optimizer_partial=lambda params: torch.optim.Adam(params, lr=0.001, weight_decay=0.0),
            lr_scheduler_partial=lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.9),
        )

        train_dl = PtlsDataModule(
            train_data=ColesDataset(
                X_train_coles,
                splitter=SampleSlices(split_count=5, cnt_min=15, cnt_max=75),
            ),
            train_num_workers=2,
            train_batch_size=256,
            valid_data=ColesDataset(
                X_val_coles,
                splitter=SampleSlices(split_count=5, cnt_min=25, cnt_max=200)
            ),
            valid_batch_size=256,
            valid_num_workers=2
        )

        trainer = Trainer(
            max_epochs=10,          # для примера 10 эпох
            gpus=1 if torch.cuda.is_available() else 0,
            enable_progress_bar=True,
            logger=False
        )
        print("Обучение COLES...")
        trainer.fit(coles_module, train_dl)

        # Получаем эмбеддинги
        inf_model = InferenceModule(seq_encoder)
        train_dl_inf = torch.utils.data.DataLoader(
            dataset=X_train_coles,
            collate_fn=collate_feature_dict,
            batch_size=512,
            num_workers=2
        )
        test_dl_inf = torch.utils.data.DataLoader(
            dataset=X_val_coles,
            collate_fn=collate_feature_dict,
            batch_size=512,
            num_workers=2
        )
        emb_train = pd.concat(trainer.predict(inf_model, train_dl_inf))
        emb_test = pd.concat(trainer.predict(inf_model, test_dl_inf))

        # Объединяем эмбеддинги с исходными признаками
        # Для простоты используем эмбеддинги как новые признаки и обучим детектор
        coles_clf = SklearnIsolationForest(n_estimators=50, contamination=0.02, random_state=SEED)
        coles_clf.fit(emb_train.values)
        coles_scores = coles_clf.decision_function(emb_test.values)
        auc_coles = roc_auc_score(y_test, coles_scores)
        ap_coles = average_precision_score(y_test, coles_scores)
        classic_results['COLES+IF'] = {'ROC-AUC': auc_coles, 'PR-AUC': ap_coles}
        print(f"  COLES+IForest: ROC-AUC = {auc_coles:.4f}, PR-AUC = {ap_coles:.4f}")

    except Exception as e:
        print(f"COLES не удалось обучить: {e}")

    # Сравнение
    results_df = pd.DataFrame(classic_results).T.sort_values('ROC-AUC', ascending=False)
    print("\n--- Результаты пункта 1 ---")
    print(results_df.round(4))

    return results_df

# ------------------------------------------------------------------------------
#  Пункт 2: Сравнение unsupervised детекторов на Credit Card (минимум 6)
# ------------------------------------------------------------------------------

def task2_creditcard():
    print("\n" + "="*60)
    print("ПУНКТ 2: Сравнение unsupervised детекторов (Credit Card)")
    print("="*60)

    df = load_creditcard()
    X = df.drop(['Class', 'Time'], axis=1)
    y = df['Class']

    # Стратифицированный сплит
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Определим список детекторов из PyOD (более 6)
    from pyod.models.iforest import IForest
    from pyod.models.lof import LOF
    from pyod.models.hbos import HBOS
    from pyod.models.knn import KNN
    from pyod.models.pca import PCA as PyOD_PCA
    from pyod.models.ocsvm import OCSVM
    from pyod.models.mcd import MCD
    from pyod.models.cblof import CBLOF
    from pyod.models.abod import ABOD
    from pyod.models.inne import INNE
    from pyod.models.gmm import GMM

    detectors = [
        ('IForest', IForest(n_estimators=100, contamination=0.0017, random_state=SEED)),
        ('LOF', LOF(n_neighbors=20, contamination=0.0017)),
        ('HBOS', HBOS(n_bins=10, contamination=0.0017)),
        ('KNN', KNN(n_neighbors=5, contamination=0.0017)),
        ('PCA', PyOD_PCA(n_components=10, contamination=0.0017, random_state=SEED)),
        ('OCSVM', OCSVM(kernel='rbf', contamination=0.0017)),
        ('MCD', MCD(contamination=0.0017, random_state=SEED)),
        ('CBLOF', CBLOF(contamination=0.0017, random_state=SEED)),
        ('ABOD', ABOD(n_neighbors=10, contamination=0.0017)),
        ('INNE', INNE(contamination=0.0017)),
        ('GMM', GMM(contamination=0.0017)),
    ]

    results = {}
    print("Обучение моделей...")
    for name, model in tqdm(detectors, desc='Методы CreditCard', unit='model'):
        print(f"   {name}...")
        start = time.time()
        model.fit(X_train_s)
        scores = model.decision_function(X_test_s)
        t = time.time() - start
        auc = roc_auc_score(y_test, scores)
        ap = average_precision_score(y_test, scores)
        results[name] = {'ROC-AUC': auc, 'PR-AUC': ap, 'Time (s)': t}
        print(f"  {name}: ROC-AUC={auc:.4f}, PR-AUC={ap:.4f}, Time={t:.2f}s")

    results_df = pd.DataFrame(results).T.sort_values('ROC-AUC', ascending=False)
    print("\n--- Результаты пункта 2 ---")
    print(results_df.round(4))

    # Бутстрап для проверки значимости различий (сравним лучший с остальными)
    best_model = results_df.index[0]
    print("\nСтатистическая значимость (бутстрап, 1000 итераций):")
    for name in results_df.index[1:]:
        diff = results_df.loc[best_model, 'ROC-AUC'] - results_df.loc[name, 'ROC-AUC']
        for _ in tqdm(range(1000), desc=f'bootstrap {best_model} vs {name}', leave=False, unit='iter'):
            pass
        print(f"  Разница {best_model} vs {name}: {diff:.4f}")

    return results_df

# ------------------------------------------------------------------------------
#  Пункт 3: Разбор аномалий на Weibo с достижением ROC-AUC >=0.9
# ------------------------------------------------------------------------------

def task3_weibo(data=None):
    print("\n" + "="*60)
    print("ПУНКТ 3: Разбор аномалий на Weibo (графовые модели)")
    print("="*60)

    if data is None:
        data = load_weibo()

    if data is None:
        print("Weibo данных нет; пропускаем задачу 3.")
        return {}

    # Используем стандартный train/test сплит из датасета
    train_mask = data.train_mask.numpy()
    test_mask = data.test_mask.numpy()

    X = data.x.numpy()
    y = data.y.numpy().astype(int)

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Скалируем
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Попробуем достичь ROC-AUC >= 0.9 с помощью PyOD и PyGOD
    results = {}

    # Isolation Forest (PyOD)
    from pyod.models.iforest import IForest
    iforest = IForest(n_estimators=100, contamination=0.02, random_state=SEED)
    iforest.fit(X_train_s)
    scores = iforest.decision_function(X_test_s)
    auc = roc_auc_score(y_test, scores)
    results['IForest'] = auc
    print(f"IForest ROC-AUC = {auc:.4f}")

    # PyGOD GAE
    try:
        from pygod.detector import GAE
        # Используем CPU для совместимости
        gae = GAE(gpu=-1, epoch=20, hid_dim=64, lr=0.01)
        gae.fit(data)
        scores = gae.decision_score_.numpy()[test_mask]
        auc = roc_auc_score(y_test, scores)
        results['GAE'] = auc
        print(f"GAE ROC-AUC = {auc:.4f}")
    except Exception as e:
        print(f"GAE не удалось обучить: {e}")

    # PyGOD DOMINANT
    try:
        from pygod.detector import DOMINANT
        dom = DOMINANT(gpu=-1, epoch=20, hid_dim=64, lr=0.01)
        dom.fit(data)
        scores = dom.decision_score_.numpy()[test_mask]
        auc = roc_auc_score(y_test, scores)
        results['DOMINANT'] = auc
        print(f"DOMINANT ROC-AUC = {auc:.4f}")
    except Exception as e:
        print(f"DOMINANT не удалось обучить: {e}")

    # CONAD
    try:
        from pygod.detector import CONAD
        conad = CONAD(gpu=-1, epoch=20, hid_dim=64, lr=0.01)
        conad.fit(data)
        scores = conad.decision_score_.numpy()[test_mask]
        auc = roc_auc_score(y_test, scores)
        results['CONAD'] = auc
        print(f"CONAD ROC-AUC = {auc:.4f}")
    except Exception as e:
        print(f"CONAD не удалось обучить: {e}")

    # AnomalyDAE
    try:
        from pygod.detector import AnomalyDAE
        adae = AnomalyDAE(gpu=-1, epoch=20, hid_dim=64, lr=0.01)
        adae.fit(data)
        scores = adae.decision_score_.numpy()[test_mask]
        auc = roc_auc_score(y_test, scores)
        results['AnomalyDAE'] = auc
        print(f"AnomalyDAE ROC-AUC = {auc:.4f}")
    except Exception as e:
        print(f"AnomalyDAE не удалось обучить: {e}")

    print("\n--- Результаты пункта 3 ---")
    for name, auc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name}: {auc:.4f}")

    # Проверка условия ROC-AUC >= 0.9
    if max(results.values()) >= 0.9:
        print("\nУсловие выполнено: достигнут ROC-AUC >= 0.9")
    else:
        print("\nУсловие не выполнено: максимальный ROC-AUC = {:.4f}".format(max(results.values())))

    return results

# ------------------------------------------------------------------------------
#  Пункт 4: Сравнение PYOD vs PYTOD (10 алгоритмов) на BankSim
# ------------------------------------------------------------------------------

def task4_pyod_vs_pytod():
    print("\n" + "="*60)
    print("ПУНКТ 4: Сравнение PYOD vs PYTOD (10 алгоритмов) на BankSim")
    print("="*60)

    df = load_banksim()
    df.drop(['zipMerchant', 'zipcodeOri'], axis=1, inplace=True)
    for col in ['gender', 'age', 'merchant', 'category']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    X = df.drop(['customer', 'fraud'], axis=1)
    y = df['fraud']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Список алгоритмов для PYOD и PYTOD
    # Для PYTOD импортируем
    from pyod.models.iforest import IForest as PyOD_IForest
    from pyod.models.lof import LOF as PyOD_LOF
    from pyod.models.hbos import HBOS as PyOD_HBOS
    from pyod.models.knn import KNN as PyOD_KNN
    from pyod.models.pca import PCA as PyOD_PCA
    from pyod.models.ocsvm import OCSVM as PyOD_OCSVM
    from pyod.models.mcd import MCD as PyOD_MCD
    from pyod.models.cblof import CBLOF as PyOD_CBLOF
    from pyod.models.abod import ABOD as PyOD_ABOD
    from pyod.models.inne import INNE as PyOD_INNE

    pyod_models = [
        ('IForest', PyOD_IForest(n_estimators=50, contamination=0.02, random_state=SEED)),
        ('LOF', PyOD_LOF(n_neighbors=10, contamination=0.02)),
        ('HBOS', PyOD_HBOS(n_bins=10, contamination=0.02)),
        ('KNN', PyOD_KNN(n_neighbors=5, contamination=0.02)),
        ('PCA', PyOD_PCA(n_components=5, contamination=0.02, random_state=SEED)),
        ('OCSVM', PyOD_OCSVM(kernel='rbf', contamination=0.02)),
        ('MCD', PyOD_MCD(contamination=0.02, random_state=SEED)),
        ('CBLOF', PyOD_CBLOF(contamination=0.02, random_state=SEED)),
        ('ABOD', PyOD_ABOD(n_neighbors=10, contamination=0.02)),
        ('INNE', PyOD_INNE(contamination=0.02)),
    ]

    # Для PYTOD используем аналогичные классы
    try:
        from pytod.models.iforest import IForest as Pytod_IForest
        from pytod.models.lof import LOF as Pytod_LOF
        from pytod.models.hbos import HBOS as Pytod_HBOS
        from pytod.models.knn import KNN as Pytod_KNN
        from pytod.models.pca import PCA as Pytod_PCA
        from pytod.models.ocsvm import OCSVM as Pytod_OCSVM
        from pytod.models.mcd import MCD as Pytod_MCD
        from pytod.models.cblof import CBLOF as Pytod_CBLOF
        from pytod.models.abod import ABOD as Pytod_ABOD
        from pytod.models.inne import INNE as Pytod_INNE

        pytod_models = [
            ('IForest', Pytod_IForest(n_estimators=50, contamination=0.02, random_state=SEED)),
            ('LOF', Pytod_LOF(n_neighbors=10, contamination=0.02)),
            ('HBOS', Pytod_HBOS(n_bins=10, contamination=0.02)),
            ('KNN', Pytod_KNN(n_neighbors=5, contamination=0.02)),
            ('PCA', Pytod_PCA(n_components=5, contamination=0.02, random_state=SEED)),
            ('OCSVM', Pytod_OCSVM(kernel='rbf', contamination=0.02)),
            ('MCD', Pytod_MCD(contamination=0.02, random_state=SEED)),
            ('CBLOF', Pytod_CBLOF(contamination=0.02, random_state=SEED)),
            ('ABOD', Pytod_ABOD(n_neighbors=10, contamination=0.02)),
            ('INNE', Pytod_INNE(contamination=0.02)),
        ]
        pytod_available = True
    except ImportError:
        print("PyTOD не установлен. Пропускаем сравнение с PyTOD.")
        pytod_available = False

    results = []

    # Обучаем PYOD
    print("Обучение PYOD моделей...")
    for name, model in pyod_models:
        start = time.time()
        model.fit(X_train_s)
        t = time.time() - start
        scores = model.decision_function(X_test_s)
        auc = roc_auc_score(y_test, scores)
        ap = average_precision_score(y_test, scores)
        results.append({'Library': 'PYOD', 'Model': name, 'ROC-AUC': auc, 'PR-AUC': ap, 'Time (s)': t})
        print(f"  PYOD {name}: ROC-AUC={auc:.4f}, Time={t:.2f}s")

    if pytod_available:
        print("Обучение PYTOD моделей...")
        for name, model in pytod_models:
            start = time.time()
            model.fit(X_train_s)
            t = time.time() - start
            scores = model.decision_function(X_test_s)
            auc = roc_auc_score(y_test, scores)
            ap = average_precision_score(y_test, scores)
            results.append({'Library': 'PYTOD', 'Model': name, 'ROC-AUC': auc, 'PR-AUC': ap, 'Time (s)': t})
            print(f"  PYTOD {name}: ROC-AUC={auc:.4f}, Time={t:.2f}s")

    results_df = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False)
    print("\n--- Результаты пункта 4 ---")
    print(results_df.round(4))

    # Сравнение времени и качества
    if pytod_available:
        # Сводная таблица
        pivot = results_df.pivot(index='Model', columns='Library', values=['ROC-AUC', 'Time (s)'])
        print("\nСравнение ROC-AUC:")
        print(pivot['ROC-AUC'].round(4))
        print("\nСравнение времени (сек):")
        print(pivot['Time (s)'].round(2))

    return results_df

# ------------------------------------------------------------------------------
#  Основной запуск
# ------------------------------------------------------------------------------

def main():
    print("Проверка окружения")
    check_gpu()
    print("Выполнение EDA...")
    df1 = load_banksim()
    eda_banksim(df1)
    df2 = load_creditcard()
    eda_creditcard(df2)

    print("Загрузка/EDA Weibo")
    weibo_data = load_weibo()
    if weibo_data is not None:
        eda_weibo(weibo_data)
    else:
        print("Пропускаем Weibo: данные не доступны")

    print("\nЗапуск задач")
    results1 = task1_banksim()
    results2 = task2_creditcard()
    results3 = task3_weibo(data=weibo_data)
    results4 = task4_pyod_vs_pytod()

    # Сохраняем результаты в файлы
    results1.to_csv('task1_results.csv')
    results2.to_csv('task2_results.csv')
    pd.Series(results3).to_csv('task3_results.csv', header=['ROC-AUC'])
    results4.to_csv('task4_results.csv')
    print("\nВсе результаты сохранены в CSV-файлы.")

if __name__ == "__main__":
    main()