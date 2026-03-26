import urllib.request
import zipfile
import pandas as pd

from antifraud.data_module.base import DatasetLoader


class CreditCardLoader(DatasetLoader):
    def load(self, data_root) -> pd.DataFrame:
        data_dir = data_root / 'creditcard_data'
        data_dir.mkdir(exist_ok=True)

        csv_path = data_dir / 'creditcard.csv'
        zip_path = data_dir / 'creditcard.zip'

        url = 'https://github.com/jeffprosise/Machine-Learning/raw/master/Data/creditcard.zip'

        if csv_path.exists():
            return pd.read_csv(csv_path)

        if not zip_path.exists():
            print(f'Загрузка {url} -> {zip_path}')
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'python-urllib'})
                with urllib.request.urlopen(req, timeout=30) as response:
                    with open(zip_path, 'wb') as f:
                        while True:
                            chunk = response.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)
            except Exception as err:
                raise RuntimeError(
                    'Не удалось скачать creditcard.zip. '
                    'Скачайте вручную и поместите в creditcard_data/ или проверьте прокси.'
                ) from err

        # Validate zip integrity and retry once
        def _is_valid_zip(path):
            try:
                with zipfile.ZipFile(path, 'r') as z:
                    return z.testzip() is None
            except zipfile.BadZipFile:
                return False

        if not _is_valid_zip(zip_path):
            zip_path.unlink(missing_ok=True)
            print('ZIP файл поврежден. Перекачиваем...')
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'python-urllib'})
                with urllib.request.urlopen(req, timeout=30) as response:
                    with open(zip_path, 'wb') as f:
                        while True:
                            chunk = response.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)
            except Exception as err:
                raise RuntimeError('Не удалось загрузить валидный creditcard.zip') from err

        if not _is_valid_zip(zip_path):
            raise RuntimeError('Загруженный creditcard.zip повреждён. Повторно скачайте вручную.')

        with zipfile.ZipFile(zip_path, 'r') as archive:
            archive.extractall(data_dir)

        if not csv_path.exists():
            raise FileNotFoundError('creditcard.csv не найден после распаковки')

        return pd.read_csv(csv_path)
