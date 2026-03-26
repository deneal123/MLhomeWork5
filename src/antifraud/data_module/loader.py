import urllib.request
from pathlib import Path

from antifraud.data_module.base import ProxyConfig
from antifraud.data_module.banksim import BankSimLoader
from antifraud.data_module.creditcard import CreditCardLoader
from antifraud.data_module.weibo import WeiboLoader
from antifraud.config import settings


class DataLoader:
    def __init__(self, proxy_config=None):
        self.proxy_config = proxy_config or ProxyConfig()
        self.proxy_config.apply()
        self.banksim_loader = BankSimLoader()
        self.creditcard_loader = CreditCardLoader()
        self.weibo_loader = WeiboLoader()

        data_cfg = settings.get('data', None)
        outputs_cfg = settings.get('outputs', None)

        if isinstance(data_cfg, dict):
            data_root_value = data_cfg.get('root', 'data')
        else:
            data_root_value = getattr(data_cfg, 'root', 'data') if data_cfg is not None else 'data'

        if isinstance(outputs_cfg, dict):
            output_root_value = outputs_cfg.get('root', 'outputs')
        else:
            output_root_value = getattr(outputs_cfg, 'root', 'outputs') if outputs_cfg is not None else 'outputs'

        self.data_root = Path(data_root_value)
        self.output_root = Path(output_root_value)

        self.data_root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def load_banksim(self):
        return self.banksim_loader.load(self.data_root)

    def load_creditcard(self):
        return self.creditcard_loader.load(self.data_root)

    def load_weibo(self):
        return self.weibo_loader.load(self.data_root)

    def output_path(self, filename: str) -> Path:
        return self.output_root / filename

    def download_creditcard_zip(self, url: str, destination: Path):
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            return
        urllib.request.urlretrieve(url, str(destination))
