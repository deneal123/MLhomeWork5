import os
import urllib.request
import urllib.parse
import subprocess
from abc import ABC, abstractmethod


class ProxyConfig:
    def __init__(self):
        self.proxies = {}
        self._load_env()

    def _load_env(self):
        if os.environ.get('NO_PROXY', '').strip().lower() in ('1', 'true', 'yes') or \
           os.environ.get('DISABLE_PROXY', '').strip().lower() in ('1', 'true', 'yes') or \
           os.environ.get('SKIP_PROXY', '').strip().lower() in ('1', 'true', 'yes'):
            self.proxies = {}
            os.environ.pop('http_proxy', None)
            os.environ.pop('https_proxy', None)
            os.environ.pop('HTTP_PROXY', None)
            os.environ.pop('HTTPS_PROXY', None)
            return

        for scheme in ('http', 'https'):
            for key in (f'{scheme}_proxy', f'{scheme.upper()}_PROXY'):
                value = os.environ.get(key)
                if value:
                    self.proxies[scheme] = value
                    break

        if not self.proxies:
            self.proxies = urllib.request.getproxies() or {}

    def apply(self):
        if not self.proxies:
            return

        handlers = [urllib.request.ProxyHandler(self.proxies)]
        for proxy in self.proxies.values():
            parsed = urllib.parse.urlparse(proxy)
            if parsed.username and parsed.password:
                auth = urllib.request.ProxyBasicAuthHandler()
                auth.add_password(
                    realm=None,
                    uri=f"{parsed.scheme}://{parsed.hostname}",
                    user=urllib.parse.unquote(parsed.username),
                    passwd=urllib.parse.unquote(parsed.password),
                )
                handlers.append(auth)
                break

        opener = urllib.request.build_opener(*handlers)
        urllib.request.install_opener(opener)

        # also configure environment and git for subprocess operations
        if 'http' in self.proxies:
            os.environ['http_proxy'] = self.proxies['http']
            os.environ['HTTP_PROXY'] = self.proxies['http']
            subprocess.run(['git', 'config', '--global', 'http.proxy', self.proxies['http']], check=False)
        if 'https' in self.proxies:
            os.environ['https_proxy'] = self.proxies['https']
            os.environ['HTTPS_PROXY'] = self.proxies['https']
            subprocess.run(['git', 'config', '--global', 'https.proxy', self.proxies['https']], check=False)


class DatasetLoader(ABC):
    @abstractmethod
    def load(self):
        raise NotImplementedError
