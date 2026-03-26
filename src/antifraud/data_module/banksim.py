from pathlib import Path
import os
import pandas as pd
from subprocess import run

from antifraud.data_module.base import DatasetLoader


class BankSimLoader(DatasetLoader):
    def load(self, data_root):
        repo_dir = data_root / 'fraud-detection-on-banksim-data'
        if not repo_dir.exists():
            clone = os.environ.get('BANKSIM_CLONE', 'false').lower() in ('1', 'true', 'yes')
            if clone:
                run(['git', 'clone', 'https://github.com/atavci/fraud-detection-on-banksim-data.git', str(repo_dir)], check=True)
            else:
                raise FileNotFoundError(
                    f"BankSim dataset not found in {repo_dir}. "
                    "Установите BANKSIM_CLONE=true или клонируйте репозиторий вручную."
                )

        data_path = repo_dir / 'Data' / 'synthetic-data-from-a-financial-payment-system' / 'bs140513_032310.csv'
        if not data_path.exists():
            raise FileNotFoundError(f"BankSim CSV не найден по пути {data_path}")

        return pd.read_csv(data_path)
